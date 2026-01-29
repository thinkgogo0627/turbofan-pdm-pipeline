import kagglehub
import shutil
import os
from pathlib import Path

# ==========================================
# 설정 (Path Config)
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

# Kaggle Dataset ID (가장 안정적인 버전)
DATASET_HANDLE = "behrad3d/nasa-cmaps"

def ingest_data():
    print(f"[Start] Ingesting data via KaggleHub...")

    # 1. KaggleHub로 다운로드 (자동으로 ~/.cache/kagglehub 에 저장됨)
    print(f" - Downloading dataset: {DATASET_HANDLE}")
    try:
        cache_path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f" - Downloaded to cache: {cache_path}")
    except Exception as e:
        print(f"[Error] KaggleHub download failed: {e}")
        return

    # 2. 우리 프로젝트 폴더(data/raw) 생성
    if not RAW_DATA_DIR.exists():
        os.makedirs(RAW_DATA_DIR)
        print(f" - Created directory: {RAW_DATA_DIR}")

    # 3. 캐시 폴더에서 파일 찾아서 복사해오기 (Move files)
    print(" - Copying files to local data/raw directory...")
    
    file_count = 0
    # 캐시 폴더를 뒤져서 .txt 파일만 가져옴
    for root, dirs, files in os.walk(cache_path):
        for file in files:
            if file.endswith(".txt") or file.endswith(".csv"):
                src_file = os.path.join(root, file)
                dst_file = RAW_DATA_DIR / file
                
                # 복사 (이미 있으면 덮어쓰기)
                shutil.copy2(src_file, dst_file)
                print(f"   > Copied: {file}")
                file_count += 1

    print("-" * 30)
    if file_count > 0:
        print(f"[Done] Successfully ingested {file_count} files to {RAW_DATA_DIR}")
        print("Ready for Feature Engineering!")
    else:
        print("[Warning] No files found in the downloaded dataset.")
    print("-" * 30)

if __name__ == "__main__":
    ingest_data()