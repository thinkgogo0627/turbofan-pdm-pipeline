import os
import urllib.request
from pathlib import Path

# 프로젝트 루트 및 데이터 저장 경로
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "raw"

# 절대 터지지 않는 GitHub Raw 파일 경로 (가장 안정적인 미러)
BASE_URL = "https://raw.githubusercontent.com/ankurchourasia/CMAPSSData/master/"

# 우리가 필요한 파일 리스트 (이 파일들이 있어야 분석 가능)
FILES_TO_DOWNLOAD = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
    "readme.txt"
]

def download_data():
    # 1. 저장할 폴더 생성
    if not DATA_DIR.exists():
        os.makedirs(DATA_DIR)
        print(f"[Info] Created directory: {DATA_DIR}")

    print(f"[Start] Downloading files to {DATA_DIR}...")
    
    # 2. 파일 하나씩 순회하며 다운로드
    for filename in FILES_TO_DOWNLOAD:
        file_path = DATA_DIR / filename
        file_url = BASE_URL + filename
        
        if not file_path.exists():
            try:
                print(f" - Downloading {filename}...", end=" ")
                urllib.request.urlretrieve(file_url, file_path)
                print("Done.")
            except Exception as e:
                print(f"\n[Error] Failed to download {filename}: {e}")
        else:
            print(f" - {filename} already exists. Skipping.")

    # 3. 결과 확인
    print("-" * 30)
    print("Download Complete. File list:")
    files = sorted(os.listdir(DATA_DIR))
    for f in files:
        print(f" > {f}")
    print(f"Total files: {len(files)}")
    print("-" * 30)

if __name__ == "__main__":
    download_data()