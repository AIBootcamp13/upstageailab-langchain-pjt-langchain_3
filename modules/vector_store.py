# vector_store.py
import json
import os
from langchain_community.vectorstores import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.docstore.document import Document
from . import config
from dotenv import load_dotenv

# Using built-in sqlite3 instead of pysqlite3 to avoid import issues
import sqlite3

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class VectorStoreManager:
    """
    전처리된 데이터를 로드하여 벡터 DB를 구축하고 관리하는 클래스.
    """
    def __init__(self, persist_directory=config.CHROMA_DB_PATH):
        self.persist_directory = persist_directory
        self.doc_embedding = UpstageEmbeddings(
            api_key=api_key,
            model="solar-embedding-1-large-passage"
)
        self.query_embedding = UpstageEmbeddings(
            api_key=api_key,
            model="solar-embedding-1-large-query"
)

    
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

    def build(self, json_path=config.MERGED_PREPROCESSED_FILE):
        print("INFO: 벡터 DB 구축을 시작합니다...")
        
        if not os.path.exists(json_path):
            print(f"ERROR: '{json_path}' 파일이 존재하지 않습니다.")
            return None

        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        documents = []
        for item in data:
            metadata = {
                'id': item.get('id', ''),
                'title': item.get('title', ''),
                'ingredients': item.get('ingredients', ''),
                'url': item.get('url', '')
            }
            doc = Document(page_content=item.get('combined_text', ''), metadata=metadata)
            documents.append(doc)
        
        print(f"INFO: 총 {len(data)}개의 문서를 {len(documents)}개의 청크로 분할했습니다.")
        
        # Chroma 벡터스토어 구축
        print("INFO: 'passage' 모델로 문서 임베딩 및 DB 저장을 진행합니다.")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.doc_embedding,
            persist_directory=self.persist_directory
        )
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