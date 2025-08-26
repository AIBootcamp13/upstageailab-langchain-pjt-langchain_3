# main.py
import os
import argparse
# --- 수정된 부분: 모든 모듈을 'modules' 폴더에서 가져오도록 변경 ---
from modules import config
from modules.crawler import RecipeCrawler
from modules.preprocess import DataPreprocessor
from modules.vector_store import VectorStoreManager
from modules.retriever import AdvancedRetriever
from modules.llm_handler import LLMHandler
from langchain.storage import InMemoryStore # --- 추가된 부분 ---

from modules.utils_docstore import register_parent_docs

# --- 추가/수정된 부분 ---
def main(rebuild_db: bool, until_step: str):
    """
    QA 엔진의 전체 실행 흐름을 제어하는 메인 함수.
    """
    # 1. 크롤링
    if not os.path.exists(config.CRAWLED_DATA_DIR) or not os.listdir(config.CRAWLED_DATA_DIR):
        print("--- 1. 데이터 크롤링 시작 ---")
        os.makedirs(config.CRAWLED_DATA_DIR, exist_ok=True)
        crawler = RecipeCrawler()
        crawler.run(start_page=1, end_page=3, output_filename=os.path.join(config.CRAWLED_DATA_DIR, "baek_recipes_1-3.json"))
        crawler.run(start_page=11, end_page=20, output_filename=os.path.join(config.CRAWLED_DATA_DIR, "baek_recipes_11-20.json"))
        crawler.run(start_page=21, end_page=30, output_filename=os.path.join(config.CRAWLED_DATA_DIR, "baek_recipes_21-30.json"))
        crawler.run(start_page=31, end_page=40, output_filename=os.path.join(config.CRAWLED_DATA_DIR, "baek_recipes_31-40.json"))
        crawler.run(start_page=41, end_page=50, output_filename=os.path.join(config.CRAWLED_DATA_DIR, "baek_recipes_41-50.json"))
        crawler.run(start_page=51, end_page=60, output_filename=os.path.join(config.CRAWLED_DATA_DIR, "baek_recipes_51-60.json"))
    else:
        print(f"--- 1. 크롤링 건너뛰기 ('{config.CRAWLED_DATA_DIR}' 폴더에 파일이 이미 존재합니다) ---")

    # 'crawl' 단계까지만 실행하는 옵션 확인
    if until_step == 'crawl':
        print("\nSUCCESS: 'crawl' 단계까지 실행이 완료되었습니다.")
        return

    # 2. 데이터 전처리
    preprocessing_needed = not os.path.exists(config.MERGED_PREPROCESSED_FILE)
    if preprocessing_needed:
        print("\n--- 2. 데이터 전처리 시작 ---")
        preprocessor = DataPreprocessor()
        success = preprocessor.run(config.CRAWLED_DATA_DIR, config.MERGED_PREPROCESSED_FILE)
        
        if not success:
            print("CRITICAL: 데이터 전처리에 실패하여 프로그램을 종료합니다.")
            return
    else:
        print(f"--- 2. 전처리 건너뛰기 ('{config.MERGED_PREPROCESSED_FILE}' 파일이 이미 존재합니다) ---")

    # 'preprocess' 단계까지만 실행하는 옵션 확인
    if until_step == 'preprocess':
        print("\nSUCCESS: 'preprocess' 단계까지 실행이 완료되었습니다.")
        return

    # --- 이하 코드는 until_step == 'run' 일 때만 실행됩니다. ---
    
    # 3. 벡터 DB 구축 또는 로드
    print("\n--- 3. 벡터 DB 준비 시작 ---")
    vs_manager = VectorStoreManager()
    
    # --- 수정된 부분: DB 구축 시 ParentDocumentRetriever를 위한 준비를 함께 진행 ---
    docstore = InMemoryStore() # 부모 문서를 메모리에 저장할 공간 생성

    if rebuild_db or not os.path.exists(config.CHROMA_DB_PATH):
        if rebuild_db: print("INFO: --rebuild-db 옵션에 따라 DB를 새로 구축합니다.")
        # build 함수에 docstore를 넘겨주어 부모-자식 문서를 함께 처리하도록 함
        vectorstore = vs_manager.build(docstore=docstore, json_path=config.MERGED_PREPROCESSED_FILE)
    else:
        # DB를 로드할 때는, 원본 문서를 읽어와서 docstore를 채워줘야 함
        # (InMemoryStore는 휘발성이므로 프로그램을 켤 때마다 채워야 함)
        vectorstore = vs_manager.load()
        parent_documents = vs_manager._load_documents_from_json(config.MERGED_PREPROCESSED_FILE)
        register_parent_docs(docstore, parent_documents) # docstore에 부모 문서 저장

    if not vectorstore:
        print("CRITICAL: 벡터 DB 준비에 실패하여 프로그램을 종료합니다.")
        return

    # 4. Advanced RAG 리트리버 설정
    print("\n--- 4. RAG 리트리버 설정 ---")
    # --- 수정된 부분: retriever에 docstore를 넘겨주고, add_documents 호출 삭제! ---
    adv_retriever = AdvancedRetriever(vectorstore, docstore)
    retriever = adv_retriever.get_retriever()
    print("INFO: ParentDocumentRetriever 설정 완료.")
    # retriever.add_documents(docs_for_retriever) # 👈 문제가 됐던 이 라인을 삭제!
    
    # 5. LLM 핸들러 및 RAG 체인 생성
    print("\n--- 5. QA 엔진(LLM) 초기화 ---")
    llm_handler = LLMHandler(retriever=retriever)
    qa_chain = llm_handler.create_rag_chain()
    print("SUCCESS: 백종원 레시피 QA 엔진이 준비되었습니다!")

    # 6. 대화형 QA 세션 시작
    print("\n--- 안녕하세요! 백주부입니다. 뭐든 물어보셔유 (종료하려면 '그만') ---")
    session_id = "user_session_01" 
    
    while True:
        try:
            user_input = input("🤔 질문: ")
            if user_input.lower() == '그만':
                print("\n다음에 또 찾아주셔유! 맛있게 해드세유~")
                break
            
            response = qa_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            print("\n백주부 💬:")
            print(response['answer'])
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n다음에 또 찾아주셔유! 맛있게 해드세유~")
            break
        except Exception as e:
            print(f"\nERROR: 죄송해유, 처리 중에 문제가 생겼어유: {e}")

if __name__ == '__main__':
    # --- 추가/수정된 부분: 실행 옵션 추가 ---
    parser = argparse.ArgumentParser(description="백종원 레시피 QA 엔진")
    parser.add_argument(
        '--rebuild-db',
        action='store_true',
        help="기존 벡터 DB를 무시하고 새로 구축합니다."
    )
    parser.add_argument(
        '--until-step',
        type=str,
        choices=['crawl', 'preprocess', 'run'],
        default='run',
        help="실행할 마지막 단계를 지정합니다: crawl, preprocess, run (전체 실행)"
    )
    args = parser.parse_args()
    
    main(rebuild_db=args.rebuild_db, until_step=args.until_step)


# 가상환경 활성화 source myenv/bin/activate