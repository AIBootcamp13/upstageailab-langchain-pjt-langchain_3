# vector_store.py
import json
import os
import sys
import sqlite3
# 내장 sqlite3 모듈을 pysqlite3로 덮어쓰기
sys.modules["sqlite3"] = sqlite3
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.storage import InMemoryStore # --- 추가된 부분 ---
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from . import config
from .utils_docstore import compute_doc_id, register_parent_docs, make_child_chunks

"""
환경변수(.env):
  CHROMA_DB_PATH=./chroma_db_chunk400_punct
  CHUNK_STRATEGY=punct           # fixed | overlap | punct | section
  CHUNK_SIZE=400
  CHUNK_OVERLAP=60
  UPSTAGE_API_KEY=ls_xxx         # Upstage Embeddings
"""

# ----------------- 청킹 전략 -----------------
def splitter_fixed(size: int, overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )

def splitter_overlap(size: int, overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )

def splitter_punct(size: int, overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n[조리]\n", "\n[재료]\n", "\n\n", "\n", ". ", ", ", " ", ""],
    )

def splitter_section(size: int, overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n[조리]\n", "\n[재료]\n", "\n\n", "\n", ". ", ", ", " ", ""],
    )

_SPLIT = {
    "fixed": splitter_fixed,
    "overlap": splitter_overlap,
    "punct": splitter_punct,
    "section": splitter_section,
}

@dataclass
class VSConfig:
    db_path: str
    strategy: str
    chunk_size: int
    chunk_overlap: int

    @staticmethod
    def from_env():
        return VSConfig(
            db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db_default"),
            strategy=os.getenv("CHUNK_STRATEGY", "punct"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
            batch_size = int(os.getenv("BATCH_SIZE", "128")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "60")),
        )




class VectorStoreManager:
    """
    전처리된 데이터를 로드하여 벡터 DB를 구축하고 관리하는 클래스.
    """
    def __init__(self, persist_directory=config.CHROMA_DB_PATH):
        self.persist_directory = persist_directory
        self.doc_embedding = UpstageEmbeddings(model="solar-embedding-1-large-passage", api_key=config.UPSTAGE_API_KEY)
        self.query_embedding = UpstageEmbeddings(model="solar-embedding-1-large-query", api_key=config.UPSTAGE_API_KEY)
    
    def _load_documents_from_json(self, json_path):
        if not os.path.exists(json_path):
            print(f"WARNING: '{json_path}' 파일이 존재하지 않습니다.")
            return []
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            # combined_text를 주 내용으로, 나머지를 메타데이터로 저장
            metadata = {
                'id': item.get('id', ''),
                'title': item.get('title', ''),
                'ingredients': item.get('ingredients', ''),
                'url': item.get('url', '')
            }
            doc = Document(page_content=item.get('combined_text', ''), metadata=metadata)
            documents.append(doc)
        return documents

    # --- 👇 여기가 핵심 수정 부분입니다! (DB 구축 로직 변경) 👇 ---
    def build(self, docstore: InMemoryStore, json_path=config.MERGED_PREPROCESSED_FILE):
        print("INFO: ParentDocumentRetriever용 벡터 DB 구축을 시작합니다...")
        
        parent_documents = self._load_documents_from_json(json_path)
        if not parent_documents:
            print("ERROR: 벡터 DB를 구축할 문서가 없습니다.")
            return None

        # 부모 문서(원본 레시피)에 고유 ID를 부여하고 docstore에 저장
        register_parent_docs(docstore, parent_documents)
        # 자식 문서(잘게 쪼갠 조각) 생성
        child_documents = make_child_chunks(parent_documents, chunk_size=400, chunk_overlap=60)
        
        print(f"INFO: 총 {len(parent_documents)}개의 부모 문서를 {len(child_documents)}개의 자식 청크로 분할했습니다.")
        print("INFO: 'passage' 모델로 자식 청크 임베딩 및 DB 저장을 진행합니다.")

        batch_size = int(os.getenv("BATCH_SIZE", "128"))
        # 자식 문서를 벡터DB에 저장 (Langchain의 from_documents는 자동 배치 처리 기능이 있음)
        vectorstore = Chroma(
             persist_directory=self.persist_directory,
             embedding_function=self.doc_embedding,  # ← 빈 컬렉션 생성
        )

        total = len(child_documents)
        if total == 0:
            print("WARN: 인덱싱할 청크가 없습니다.")
        else:
            if tqdm:
                pbar = tqdm(total=total, desc="임베딩/인덱싱 진행", unit="청크")
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    vectorstore.add_documents(child_documents[start:end])
                    pbar.update(end - start)
                pbar.close()
            else:
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    vectorstore.add_documents(child_documents[start:end])

            print(f"SUCCESS: 벡터 DB 구축 완료. '{self.persist_directory}'에 저장되었습니다.")
            return vectorstore
        
    def load(self):
        if not os.path.exists(self.persist_directory):
            print("ERROR: 저장된 벡터 DB가 없습니다. 먼저 DB를 구축해야 합니다.")
            return None
            
        # --- 수정: DB 로드(쿼리) 시에는 'query' 질문용 임베딩 모델 사용 ---
        print("INFO: 기존 벡터 DB를 'query' 모델로 불러옵니다...")
        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.query_embedding # 👈 질문용 모델 사용
        )
        return vectorstore