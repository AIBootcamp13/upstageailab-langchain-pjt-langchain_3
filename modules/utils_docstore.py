# modules/utils_docstore.py
import uuid
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def compute_doc_id(md: dict) -> str:
    """
    항상 같은 입력 → 같은 doc_id 반환
    우선순위: url > (title|ingredients) > 원본 id > fallback
    """
    key = (
        md.get("url")
        or (md.get("title", "") + "|" + md.get("ingredients", "")).strip()
        or md.get("id", "")
    )
    if not key:
        key = "fallback|" + uuid.uuid4().hex
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

def register_parent_docs(docstore, parent_documents):
    """
    부모 문서에 doc_id 부여 후 docstore에 저장
    Returns:
        list[str]: 부여된 doc_id 리스트
    """
    doc_ids = []
    for doc in parent_documents:
        doc.metadata["doc_id"] = compute_doc_id(doc.metadata)
        doc_ids.append(doc.metadata["doc_id"])
    docstore.mset(list(zip(doc_ids, parent_documents)))
    return doc_ids

def make_child_chunks(parent_documents, chunk_size=400, chunk_overlap=60):
    """
    부모 문서를 잘게 쪼개서 자식 청크를 생성하고,
    각 자식의 메타데이터에 부모의 doc_id를 복사한다.
    
    Args:
        parent_documents (List[Document]): 부모 문서 리스트
        chunk_size (int): 청크 크기
        chunk_overlap (int): 청크 겹침
    
    Returns:
        List[Document]: 자식 청크 문서 리스트
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    children = []
    for p in parent_documents:
        for ch in splitter.split_documents([p]):
            ch.metadata["doc_id"] = p.metadata["doc_id"]
            children.append(ch)
    return children