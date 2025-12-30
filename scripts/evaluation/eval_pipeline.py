import argparse
import os
import sys
import subprocess
from pathlib import Path

# 출력 가독성을 위한 색상 코드
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_name):
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*60)
    print(f" [Step] {step_name}")
    print("="*60 + f"{Colors.ENDC}\n")

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f"{Colors.FAIL}[Error] File not found: {filepath}{Colors.ENDC}")
        sys.exit(1)
    return os.path.abspath(filepath)

def run_command(command, description):
    """서브프로세스로 명령어를 실행하고 에러를 처리합니다."""
    print(f"{Colors.OKBLUE}Running: {' '.join(command)}{Colors.ENDC}")
    try:
        # 실시간 출력을 위해 subprocess.run 사용
        subprocess.run(command, check=True, env=os.environ.copy())
        print(f"{Colors.OKGREEN}✔ {description} completed successfully.{Colors.ENDC}")
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}✘ Error during {description}. (Exit Code: {e.returncode}){Colors.ENDC}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Unitree Go2 Analysis Pipeline (Real Eval -> Sim2Real Comparison)")
    
    # 1. 필수 인자: 실제 로봇 로그
    parser.add_argument("--real_log", type=str, required=True, help="Path to the real-world log file (.csv or .mcap)")
    
    # 2. 필수 인자: 기존 시뮬레이션 로그 (Generation 과정 생략이므로 필수)
    parser.add_argument("--sim_log", type=str, required=True, help="Path to the existing simulation log file (.pkl or .csv)")
    
    # 3. MinIO 옵션 (선택 사항)
    parser.add_argument("--minio_endpoint", type=str, default=None, help="MinIO Endpoint URL (Optional)")
    parser.add_argument("--minio_bucket", type=str, default="robot-logs", help="MinIO Bucket Name")

    args = parser.parse_args()
    
    # 경로 절대경로로 변환 및 확인
    real_log_path = check_file_exists(args.real_log)
    sim_log_path = check_file_exists(args.sim_log)
    
    # 현재 스크립트의 위치 (다른 스크립트들도 같은 폴더에 있다고 가정)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ------------------------------------------------------------------
    # Step 1: Real-world Log Evaluation
    # ------------------------------------------------------------------
    print_step("1. Real-world Data Evaluation")
    real_eval_script = os.path.join(base_dir, "real_eval.py")
    
    if not os.path.exists(real_eval_script):
        print(f"{Colors.FAIL}[Error] real_eval.py not found in {base_dir}{Colors.ENDC}")
        sys.exit(1)

    cmd_real = [sys.executable, real_eval_script, "--real_log", real_log_path]
    run_command(cmd_real, "Real Log Analysis")

    # ------------------------------------------------------------------
    # Step 2: Simulation Log Evaluation
    # ------------------------------------------------------------------
    print_step("2. Simulation Data Evaluation")
    sim_eval_script = os.path.join(base_dir, "sim_eval.py")
    
    if os.path.exists(sim_eval_script):
        cmd_sim_analyze = [
            sys.executable, sim_eval_script,
            "--analyze_log", sim_log_path
        ]
        run_command(cmd_sim_analyze, "Sim Log Static Analysis")
    else:
        print(f"{Colors.FAIL}sim_eval.py missing.{Colors.ENDC}")
    
    # ------------------------------------------------------------------
    # Step 3: Sim-to-Real Comparison (기존 Sim 로그와 비교)
    # ------------------------------------------------------------------
    print_step("3. Sim-to-Real Comparison")
    sim2real_script = os.path.join(base_dir, "sim2real_eval.py")
    
    if not os.path.exists(sim2real_script):
        print(f"{Colors.FAIL}[Error] sim2real_eval.py not found in {base_dir}{Colors.ENDC}")
        sys.exit(1)

    cmd_sim2real = [
        sys.executable, sim2real_script,
        "--real_log", real_log_path,
        "--sim_file", sim_log_path
    ]
    
    # MinIO 옵션 전달 (sim2real_eval.py가 지원하는 경우)
    if args.minio_endpoint:
        cmd_sim2real.extend([
            "--minio_endpoint", args.minio_endpoint,
            "--minio_bucket", args.minio_bucket
        ])

    run_command(cmd_sim2real, "Sim-to-Real Comparison")

    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Analysis Pipeline Completed Successfully!{Colors.ENDC}")

if __name__ == "__main__":
    main()