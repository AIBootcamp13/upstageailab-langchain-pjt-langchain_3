# modules/config.py
import os
from dotenv import load_dotenv

# 프로젝트 루트 경로를 기준으로 .env 파일을 찾습니다.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, '.env'))

# API 키 설정 (이전과 동일)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "up_Dk6s7ks9NNd7SOKtFAO5ltetPMge0")

# --- 경로를 프로젝트 루트 기준으로 명확하게 수정 ---
CRAWLED_DATA_DIR = os.path.join(project_root, "crawled_data")
PREPROCESSED_DATA_DIR = os.path.join(project_root, "preprocessed_data")
MERGED_PREPROCESSED_FILE = os.path.join(PREPROCESSED_DATA_DIR, "all_recipes_cleaned.json")
CHROMA_DB_PATH = os.path.join(project_root, "chroma_db")