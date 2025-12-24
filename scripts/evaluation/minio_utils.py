# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
MinIO Utilities for Sim-to-Real Evaluation
------------------------------------------
Handles uploading simulation logs and downloading them for analysis.
"""

import os
import sys
import pickle
import io
import numpy as np

try:
    from minio import Minio
except ImportError:
    Minio = None

class MinioClientWrapper:
    def __init__(self, endpoint, access_key, secret_key, bucket):
        if Minio is None:
            print("[WARN] 'minio' library is missing. Install via: pip install minio")
            self.client = None
            return

        self.bucket = bucket
        try:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=False  # Assuming local/dev, change if using HTTPS
            )
            
            # Ensure bucket exists
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                print(f"[INFO] Created bucket: {self.bucket}")
                
        except Exception as e:
            print(f"[ERROR] Failed to connect to MinIO: {e}")
            self.client = None

    def upload_log(self, experiment_name, date_str, data):
        """Uploads simulation log to MinIO."""
        if self.client is None:
            print("[WARN] MinIO client not initialized. Skipping upload.")
            return False

        try:
            # Serialize
            pkl_data = pickle.dumps(data)
            pkl_stream = io.BytesIO(pkl_data)
            
            object_name = f"sim_log_{experiment_name}_{date_str}.pkl"
            
            # Upload
            self.client.put_object(
                self.bucket,
                object_name,
                pkl_stream,
                length=len(pkl_data)
            )
            print(f"[INFO] Successfully uploaded simulation log to MinIO: {self.bucket}/{object_name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to upload to MinIO: {e}")
            return False

    def download_latest_log(self, experiment_name, run_date=None):
        """Finds and downloads the latest simulation log for the given experiment."""
        if self.client is None:
            raise RuntimeError("MinIO client not initialized.")

        if not self.client.bucket_exists(self.bucket):
            raise FileNotFoundError(f"Bucket '{self.bucket}' does not exist.")

        # Search for objects
        prefix = f"sim_log_{experiment_name}"
        if run_date:
            prefix += f"_{run_date}"
            
        objects = list(self.client.list_objects(self.bucket, prefix=prefix))
        
        # Filter for .pkl files
        candidates = [obj for obj in objects if obj.object_name.endswith(".pkl")]
        
        if not candidates:
            raise FileNotFoundError(f"No logs found in MinIO for experiment: {experiment_name}")
        
        # Sort by date (descending) to get the latest
        candidates.sort(key=lambda x: x.last_modified, reverse=True)
        target_obj = candidates[0]
        
        print(f"[INFO] Downloading Sim Log: {target_obj.object_name}")
        
        response = self.client.get_object(self.bucket, target_obj.object_name)
        data = pickle.load(response)
        response.close()
        response.release_conn()
        
        return data
