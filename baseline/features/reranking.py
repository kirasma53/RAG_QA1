# RAG/baseline/features/reranking.py
import streamlit as st
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import AVAILABLE_RERANKERS, OPENAI_API_KEY
from .summarization import gpt_summarize_document # 요약 함수 import

# Reranker 모델을 로드하고 캐시
@st.cache_resource(show_spinner=False)
def get_reranker_model(method: str):
    method_lower = method.lower()
    st.write(f"Reranker 모델 로딩 중: {method_lower}...")
    if method_lower == "ko-rerank":
        return CrossEncoder("Dongjin-kr/ko-reranker", device='cpu', max_length=512)
    elif method_lower == "bge":
        return CrossEncoder("BAAI/bge-reranker-base", device='cpu', max_length=512)
    elif method_lower == "mini":
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu', max_length=512)
    else:
        raise ValueError(f"지원하지 않는 reranker method입니다: {method}")

# rerank documents
def rerank_documents(
    query: str,
    docs: List[Document],
    method: str,
    top_k: int = 3
) -> List[Tuple[Document, float]]:
    if not docs:
        return []
    if method not in AVAILABLE_RERANKERS:
        st.warning(f"지원하지 않는 리랭커({method})입니다. 리랭킹을 건너뜁니다.")
        # Return dummy scores if method is invalid or None
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)][:top_k]

    try:
        reranker = get_reranker_model(method) # Load the specified model
        st.write(f"CrossEncoder ({method})를 사용하여 문서 재순위 평가 중...")
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs, show_progress_bar=False) # st.write로 대체
        
        # scores가 단일 값일 경우 리스트로 변환 (predict 결과가 스칼라일 가능성?)
        if not isinstance(scores, list) and hasattr(scores, 'tolist'): # numpy array
            scores = scores.tolist()
        elif not isinstance(scores, list): # single float
            scores = [scores]

        scored_docs = list(zip(docs, scores))
        # Sort 분리
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
    except Exception as e:
        st.error(f"리랭킹 중 오류 발생 ({method}): {e}")
        # 오류 발생 시 원본 문서 상위 k개 반환 (점수는 0.0으로)
        return [(doc, 0.0) for doc in docs[:top_k]]

# summarize and rerank
def summarize_and_rerank(
    query: str,
    docs: List[Document],
    summarize_first: bool,
    method: Optional[str], # 리랭커 사용 안 할 수도 있으므로 Optional
    top_k: int
) -> List[Document]:
    if not docs:
        return []

    docs_to_process = docs

    if summarize_first:
        if not OPENAI_API_KEY:
            st.warning("문서 요약을 위해서는 OpenAI API 키가 필요합니다. 요약을 건너뜁니다.")
        else:
            st.write("문서 요약 중 (Reranking 전)...")
            with st.spinner("요약 진행 중... 잠시만 기다려주세요."):
                # 원본 문서의 메타데이터를 유지하면서 요약된 내용으로 Document 객체 생성
                summarized_docs_list = []
                for doc_item in docs_to_process:
                    summarized_doc = gpt_summarize_document(doc_item)
                    summarized_docs_list.append(summarized_doc)
                docs_to_process = summarized_docs_list

    final_reranked_docs: List[Document] = []
    if method and method in AVAILABLE_RERANKERS: # 리랭커 사용 및 유효한 메소드인 경우
        reranked_with_scores = rerank_documents(
            query=query,
            docs=docs_to_process, # 요약되었거나 원본인 문서들
            method=method,
            top_k=top_k
        )
        
        st.write("재순위화된 문서 (점수 순):")
        for i, (doc, score) in enumerate(reranked_with_scores):
            # 요약된 경우, 원본 내용을 메타데이터에서 가져와 최종 Document 생성
            original_content = doc.metadata.get("original_text", doc.page_content)
            final_doc = Document(page_content=original_content, metadata=doc.metadata)
            final_reranked_docs.append(final_doc)
            st.write(f"  - Rank {i+1} (Score: {score:.4f}): {final_doc.metadata.get('file_name', 'N/A')} (Index: {final_doc.metadata.get('index', 'N/A')})")
    else: # 리랭커를 사용하지 않거나 유효하지 않은 메소드인 경우
        if method: # 유효하지 않은 메소드이지만 사용하려고 한 경우 경고
             st.warning(f"리랭커 메소드 '{method}'가 유효하지 않아 리랭킹을 건너뛰고 상위 {top_k}개 문서를 사용합니다.")
        # 요약만 되었거나 원본 문서 리스트에서 top_k개 선택
        # 이 경우, 원본 내용을 가진 Document 객체로 변환할 필요가 있을 수 있음 (summarize_first=True인 경우)
        temp_docs = []
        for doc in docs_to_process[:top_k]:
            original_content = doc.metadata.get("original_text", doc.page_content)
            final_doc = Document(page_content=original_content, metadata=doc.metadata)
            temp_docs.append(final_doc)
        final_reranked_docs = temp_docs

    return final_reranked_docs