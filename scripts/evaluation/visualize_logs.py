import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob
import math

def visualize_log(log_dir, output_dir=None):
    """
    지정된 디렉토리의 모든 CSV 로그 파일을 읽어 시각화합니다.
    파일명 패턴: real_log_{YYYY-MM-DD_HH-MM-SS}_{CATEGORY}.csv
    """
    
    # 1. 파일 검색
    csv_files = glob.glob(os.path.join(log_dir, "real_log_*_*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV log files found in {log_dir}")
        return

    # 타임스탬프(Run ID)별로 파일 그룹화
    # 예: real_log_2026-01-06_13-39-47_imu.csv -> run_id: 2026-01-06_13-39-47
    runs = {}
    for fpath in csv_files:
        basename = os.path.basename(fpath)
        parts = basename.replace("real_log_", "").split("_")
        if len(parts) >= 3:
            # yyyy-mm-dd, hh-mm-ss, category.csv
            run_id = f"{parts[0]}_{parts[1]}"
            category = "_".join(parts[2:]).replace(".csv", "")
            
            if run_id not in runs:
                runs[run_id] = {}
            runs[run_id][category] = fpath

    if not runs:
        print("[WARN] Could not parse any log files.")
        return

    print(f"[INFO] Found {len(runs)} runs.")

    # 각 실행(Run)별로 처리
    for run_id, categories in runs.items():
        print(f"\n>>> Processing Run: {run_id}")
        
        # 출력 디렉토리 설정
        if output_dir:
            save_dir = os.path.join(output_dir, run_id)
        else:
            save_dir = os.path.join(log_dir, "plots", run_id)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 각 카테고리(imu, bms 등)별로 그래프 그리기
        for cat, fpath in categories.items():
            print(f"  - Visualizing {cat} ({os.path.basename(fpath)})")
            try:
                # Scan 데이터 처럼 컬럼이 너무 많은 경우(가변 길이)는 별도 처리 혹은 skip
                if cat == 'scan':
                    continue

                df = pd.read_csv(fpath)
                
                # 데이터가 비어있으면 건너뜀
                if df.empty:
                    print(f"    [SKIP] Empty file: {fpath}")
                    continue
                
                # 공통 x축 컬럼 찾기 (timestamp, wall_time 등)
                x_col = None
                for candidate in ['timestamp', 'wall_time', 'time']:
                    if candidate in df.columns:
                        x_col = candidate
                        break
                
                # x축이 없으면 인덱스 사용
                if x_col is None:
                    x_col = df.index.name if df.index.name else 'index'
                    df['index'] = df.index
                
                # 그릴 컬럼들 (x축 제외, 문자열 제외)
                plot_cols = []
                for col in df.columns:
                    if col == x_col: continue
                    if pd.api.types.is_numeric_dtype(df[col]):
                        plot_cols.append(col)
                
                if not plot_cols:
                    print(f"    [SKIP] No numeric columns to plot.")
                    continue

                # 그래프 생성 (서브플롯 활용)
                num_cols = len(plot_cols)
                cols_per_fig = 6 # 한 그림당 최대 6개 서브플롯
                num_figs = math.ceil(num_cols / cols_per_fig)
                
                for i in range(num_figs):
                    start_idx = i * cols_per_fig
                    end_idx = min((i + 1) * cols_per_fig, num_cols)
                    current_plot_cols = plot_cols[start_idx:end_idx]
                    
                    fig, axes = plt.subplots(len(current_plot_cols), 1, figsize=(10, 3 * len(current_plot_cols)), sharex=True)
                    if len(current_plot_cols) == 1:
                        axes = [axes]
                    
                    for ax, col_name in zip(axes, current_plot_cols):
                        ax.plot(df[x_col], df[col_name], label=col_name)
                        ax.set_ylabel(col_name)
                        ax.legend(loc='upper right')
                        ax.grid(True)
                    
                    axes[-1].set_xlabel(x_col)
                    fig.suptitle(f"Run: {run_id} | Category: {cat} (Part {i+1})")
                    plt.tight_layout()
                    
                    save_path = os.path.join(save_dir, f"{cat}_part{i+1}.png")
                    plt.savefig(save_path)
                    plt.close(fig)
                    print(f"    -> Saved {save_path}")

            except Exception as e:
                print(f"    [ERROR] Failed to visualize {cat}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize IsaacLab/Go2 Logs")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the logs directory containing csv files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save plots. Defaults to log_dir/plots')
    
    args = parser.parse_args()
    
    # 기본 경로가 스크립트 실행 위치 기준 상대경로일 경우 절대경로로 변환
    base_path = os.path.abspath(args.log_dir)
    
    visualize_log(base_path, args.output_dir)
