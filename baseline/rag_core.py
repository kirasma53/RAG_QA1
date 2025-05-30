# RAG/baseline/rag_core.py
import re
import streamlit as st
from typing import List, Tuple, Optional, Dict, Any
import json
import numpy as np # 평균 계산용
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from sklearn.metrics import f1_score

from config import (
    OPENAI_API_KEY, FAISS_DB_PATH, AVAILABLE_EMBEDDINGS,
    LANGSMITH_TRACING_ENABLED, LANGCHAIN_PROJECT
)
from utils import format_docs
from features.reranking import summarize_and_rerank # 리랭킹 모듈 import
from features.cot import build_user_query_prompt_chain # CoT 모듈 import

# LangSmith 설정
CALLBACKS_RAG_CORE = []
if LANGSMITH_TRACING_ENABLED:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    TRACER_RAG_CORE = LangChainTracer(project_name=f"{LANGCHAIN_PROJECT}-RAGCore")
    CALLBACKS_RAG_CORE = [TRACER_RAG_CORE]

from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langchain_openai import ChatOpenAI


@st.cache_resource(show_spinner="벡터 DB 로딩 중...")  

def load_vector_store(faiss_alias: str) -> Optional[FAISS]:
    """지정된 임베딩 모델에 대한 FAISS 벡터 저장소를 로드합니다."""
    if faiss_alias not in AVAILABLE_EMBEDDINGS:
        st.error(f"지원하지 않는 임베딩 모델 별칭입니다: {faiss_alias}")
        return None 
    
    model_name = AVAILABLE_EMBEDDINGS[faiss_alias]
    st.write(f"임베딩 모델 로딩 중: {faiss_alias} ({model_name})...")
    embedding_model = None

    if faiss_alias == "openai":
        if not OPENAI_API_KEY:
            st.error("OpenAI 임베딩 모델을 로드하려면 OpenAI API 키가 필요합니다.")
            return None
        embedding_model = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
        st.write("(OpenAI 임베딩 사용 시 API 비용이 발생할 수 있습니다)")
    else: # 로컬 모델 (HuggingFace)
        try:
            device = "cpu" # 혹은 "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True} # BGE, E5 모델 등은 True 권장
            )
        except Exception as e:
            st.error(f"HuggingFace 임베딩 모델 로딩 실패 ({model_name}): {e}")
            st.error("필요한 라이브러리가 설치되었는지 확인하세요: pip install sentence-transformers torch")
            return None

    db_path = FAISS_DB_PATH / faiss_alias
    index_file_faiss = db_path / "index.faiss"
    index_file_pkl = db_path / "index.pkl"

    if not index_file_faiss.exists() or not index_file_pkl.exists():
        st.error(f"FAISS 인덱스 파일을 찾을 수 없습니다: {db_path}")
        st.error(f"먼저 '{faiss_alias}' 모델에 대한 벡터스토어를 생성하고 해당 경로에 저장했는지 확인하세요.")
        return None

    try:
        vectorstore = FAISS.load_local(
            folder_path=str(db_path),
            embeddings=embedding_model,
            index_name="index", # 저장 시 사용한 기본 인덱스 이름
            allow_dangerous_deserialization=True # HuggingFaceEmbeddings 사용 시 필요
        )
        st.success(f"'{faiss_alias}' 벡터 DB 로드 완료!")
        return vectorstore
    except Exception as e:
        st.error(f"FAISS 벡터 DB 로드 중 오류 발생 ({faiss_alias}): {e}")
        return None


def extract_step3_only(full_response: str) -> str:
    """
    LLM이 출력한 전체 CoT 응답에서 '### 3단계:' 이후의 본문 내용만 추출.
    """
    match = re.search(r"###\s*3\s*단계[:：][^\n\r]*[\n\r]+(.*)", full_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return full_response


def run_rag_pipeline(
    question: str,
    vectorstore: FAISS,
    retriever_k: int,
    use_reranker: bool,
    reranker_method: Optional[str],
    reranker_top_k: int,
    summarize_before_rerank: bool,
    llm_model_name: str,
    use_cot: bool, # CoT 사용 여부 플래그
    run_id: str = "" # 로깅/추적용 ID,
) -> Tuple[str, List[Document]]:
    """RAG 파이프라인을 실행하여 답변과 사용된 문서를 반환합니다."""
    
    log_prefix = f"[{run_id}] " if run_id else ""
    st.write(f"{log_prefix}RAG 파이프라인 시작: 질문='{question[:50]}...'")

    # 1. Retrieve
    st.write(f"{log_prefix}1. 관련 문서 검색 중 (Retriever k={retriever_k})...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    try:
        initial_docs = retriever.invoke(question, config={"callbacks": CALLBACKS_RAG_CORE})
        st.write(f"{log_prefix}   초기 검색된 문서 {len(initial_docs)}개")
    except Exception as e:
        st.error(f"{log_prefix}문서 검색 중 오류: {e}")
        return "오류: 문서를 검색하는 중 문제가 발생했습니다.", []

    # 초기 검색 문서가 없는 경우 처리
    if not initial_docs:
        st.warning(f"{log_prefix}검색된 문서가 없습니다. 컨텍스트 없이 LLM에 질문합니다.")
        final_docs_for_llm = [] # LLM에 전달될 문서는 없음
    else:
        # 2. Rerank (Optional)
        if use_reranker and reranker_method:
            st.write(f"{log_prefix}2. 문서 재순위화 중 (Top {reranker_top_k}, Method: {reranker_method}, 요약: {summarize_before_rerank})...")
            try:
                final_docs_for_llm = summarize_and_rerank(
                    query=question,
                    docs=initial_docs,
                    summarize_first=summarize_before_rerank,
                    method=reranker_method,
                    top_k=reranker_top_k
                )
                if not final_docs_for_llm and initial_docs:
                    st.warning(f"{log_prefix}   재순위화/요약 후 유효한 문서가 없습니다. 초기 검색 결과 일부를 사용합니다.")
                    final_docs_for_llm = initial_docs[:reranker_top_k]
            except Exception as e_rerank:
                st.error(f"{log_prefix}재순위화/요약 중 오류: {e_rerank}. 초기 검색 결과 일부를 사용합니다.")
                final_docs_for_llm = initial_docs[:reranker_top_k]
        else:
            st.write(f"{log_prefix}2. 리랭커 사용 안 함. 초기 검색 결과 상위 {reranker_top_k}개 사용.")
            final_docs_for_llm = initial_docs[:reranker_top_k if reranker_top_k > 0 else len(initial_docs)]

    if not final_docs_for_llm and initial_docs: # 초기 문서는 있었으나, 리랭킹/필터링 후 비어버린 경우
        st.warning(f"{log_prefix}답변 생성에 사용할 최종 문서를 얻지 못했지만, 컨텍스트 없이 LLM에 질문합니다.")
    elif not final_docs_for_llm and not initial_docs: # 처음부터 문서가 없었던 경우
        st.warning(f"{log_prefix}답변 생성에 사용할 문서가 없습니다. 컨텍스트 없이 LLM에 질문합니다.")


    # 3. Generate (LLM 호출)
    st.write(f"{log_prefix}3. AI 답변 생성 중 ({llm_model_name}, CoT: {use_cot})... ")
    context_str = format_docs(final_docs_for_llm) if final_docs_for_llm else "참고할 문서가 없습니다."

    try:
        if use_cot:
            st.write(f"{log_prefix}   CoT 프롬프트 체인 사용...")
            cot_chain = build_user_query_prompt_chain(
                llm_model_name=llm_model_name
                # build_user_query_prompt_chain 내부에서 CALLBACKS_COT 사용
            )
            # CoT 프롬프트는 질문과 함께 컨텍스트를 활용하도록 설계 (기존 query_prompt.py 방식)
            # 질문에 컨텍스트를 포함시켜 전달
            full_question_for_cot = f"사용자 질문: {question}\n\n참고 문서:\n{context_str}"
            
            result = cot_chain.invoke(
                {"question": full_question_for_cot},
                config={"callbacks": CALLBACKS_RAG_CORE} # RAG Core의 메인 콜백 사용
            )
            
            # 결과 파싱 (AIMessage 객체 등 LangChain Core 객체일 수 있음)
            if hasattr(result, "content"):
                content = result.content
            elif isinstance(result, str):
                content = result
            elif isinstance(result, dict) and "text" in result: # 일부 모델은 dict로 반환
                content = result["text"]
            else:
                content = str(result)
            
            st.write(f"{log_prefix}   CoT 답변 생성 완료.")
            content_step3 = extract_step3_only(content)
            return content_step3, final_docs_for_llm

        else:
            st.write(f"{log_prefix}   기본 RAG 프롬프트 사용...")
            llm = ChatOpenAI(
                model_name=llm_model_name, temperature=0,
                openai_api_key=OPENAI_API_KEY, callbacks=CALLBACKS_RAG_CORE
            )
            prompt_hub = hub.pull("rlm/rag-prompt") # LangChain Hub의 기본 RAG 프롬프트
            
            rag_chain = (
                {"context": lambda _: context_str, "question": RunnablePassthrough()}
                | prompt_hub
                | llm
                | StrOutputParser()
            ).with_config({"callbacks": CALLBACKS_RAG_CORE})
            
            response = rag_chain.invoke(question)
            st.write(f"{log_prefix}   기본 RAG 답변 생성 완료.")
            return response, final_docs_for_llm

    except Exception as e:
        st.error(f"{log_prefix}LLM 호출 중 오류 발생: {e}")
        return f"오류: 답변 생성 중 문제가 발생했습니다 ({e})", final_docs_for_llm
    

def load_evaluation_set(uploaded_file, is_multiturn = False) -> Optional[List[Dict[str, Any]]]:
    """업로드된 JSON 평가셋 파일을 로드하고 유효성을 검사합니다."""
    if uploaded_file is None:
        st.error("평가셋 파일을 업로드해주세요.")
        return None

    try:
        # 파일 포인터를 처음으로 되돌림 (Streamlit의 UploadedFile 객체)
        uploaded_file.seek(0)
        eval_data = json.load(uploaded_file)

        if not isinstance(eval_data, list):
            st.error("평가셋은 JSON 리스트 형식이어야 합니다.")
            return None

        valid_data = []
        # 평가셋 항목에 필수로 있어야 하는 키 정의 (예시)
        required_keys = {'question', 'answer'} # 'p', 'num', 'no', 'name' 등도 필요시 추가

        #item의 idx와 item을 들고 온다.
        for item_idx, item in enumerate(eval_data):
            if not isinstance(item, dict):
                st.warning(f"평가셋의 {item_idx+1}번째 항목이 딕셔너리 형식이 아닙니다. 건너뜁니다.")
                continue
            
            if not required_keys.issubset(item.keys()):
                st.warning(f"평가셋의 {item_idx+1}번째 항목에 필수 키 {required_keys}가 모두 포함되어 있지 않습니다. 건너뜁니다. (현재 키: {item.keys()})")
                continue

            # 'name' 또는 'num' 또는 'M' 키가 없거나 비어있을 경우 'unrelate'로 설정 (기존 로직)
            is_name_empty = not item.get("name")
            is_num_empty = not item.get("num")
            is_M_empty = not item.get("M")
            
            #multiturn 평가셋일 때와 구분하며 기본적으로 is_multiturn은 false
            if is_multiturn : 
                if is_name_empty or is_num_empty or is_M_empty : 
                    item["name"] = ["unrelate"]
            else : 
                if is_name_empty or is_num_empty:
                    item["name"] = ["unrelate"] # 리스트 형태로 유지

            valid_data.append(item)

        if not valid_data:
            st.error("평가셋에 유효한 항목이 없습니다. 각 항목은 'question'과 'answer' 키를 포함해야 합니다.")
            return None

        st.success(f"평가셋 로드 완료: {len(valid_data)}개 질문")
        return valid_data

    except json.JSONDecodeError:
        st.error("JSON 파일 파싱 중 오류가 발생했습니다. 파일 형식을 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"평가셋 파일 처리 중 예외 발생: {e}")
        return None


def compute_evalset_f1(
    evaluation_item: Dict[str, Any],
    used_docs: List[Document]
) -> float:
    """단일 평가 항목과 사용된 문서 리스트 간의 F1 Score를 계산합니다."""
    # 파일 이름과 FAQ 번호 매핑 (필요에 따라 확장)
    file_to_faq_prefix = {
        "filtered_plants1.json": "FAQ1",
        "filtered_plants2.json": "FAQ2"
        # 추가적인 PDF 파일명 <-> 작물명 매핑이 필요하면 여기에 추가하거나 다른 방식으로 처리
    }

    # 평가 항목이 "unrelate"이면 F1 Score 계산에서 제외하거나 0점 처리
    if evaluation_item.get("name") == ["unrelate"]:
        return 0.0  # 또는 다른 값으로 처리 (예: -1로 하여 평균 계산 시 제외)

    # 평가 항목 정보 추출 (리스트일 수 있으므로 첫 번째 요소 사용, 없으면 빈 문자열)
    # evaluation_item의 'name'은 문서명 또는 FAQ 식별자, 'p'는 작물명, 'num'은 인덱스(문서 내 번호)
    ref_doc_identifier_list = evaluation_item.get("name", [])
    ref_doc_identifier = ref_doc_identifier_list[0] if ref_doc_identifier_list else ""
    
    ref_plant_list = evaluation_item.get("p", [])
    ref_plant = ref_plant_list[0] if ref_plant_list else ""
    
    raw_nums = evaluation_item.get("num", []) # 숫자 또는 숫자 리스트일 수 있음
    ref_indices = [str(n) for n in raw_nums] if isinstance(raw_nums, list) else [str(raw_nums)]
    if not any(ref_indices): # 인덱스 정보가 없는 경우
        ref_indices = []


    # 사용된 문서들의 메타데이터와 비교
    predictions = [] # 실제 RAG 시스템이 참조했다고 판단한 문서면 1, 아니면 0
    
    # 정답셋 구성: evaluation_item이 나타내는 정답 문서는 1개라고 가정
    # (실제로는 여러 개일 수 있으나, F1 계산을 위해 단순화)
    # 이 부분은 평가셋의 구조와 F1 score 계산 방식에 대한 상세 정의에 따라 달라짐.
    # 여기서는 "정답 문서를 찾았는가 (True/False)"의 형태로 F1을 계산한다고 가정.
    # 즉, y_true는 항상 [1] (정답 문서는 존재한다), y_pred는 [1] (찾음) 또는 [0] (못찾음)
    
    match_found = False
    for doc in used_docs:
        meta = doc.metadata
        doc_filename = meta.get("file_name", "") # RAG 결과 문서의 파일명
        doc_index = str(meta.get("index", ""))   # RAG 결과 문서의 인덱스
        doc_plant = meta.get("plant", "")        # RAG 결과 문서의 작물명 (PDF의 경우)

        # 1. RAG 결과 문서의 파일명이 FAQ 파일명과 매칭되는 경우 (FAQ1, FAQ2)
        if doc_filename in file_to_faq_prefix:
            # 평가셋의 name이 FAQ 식별자(FAQ1, FAQ2)와 같고,
            # 평가셋의 num(인덱스)이 RAG 결과 문서의 인덱스와 같고,
            # 평가셋의 p(작물명)가 RAG 결과 문서의 작물명과 같은지 확인
            if (file_to_faq_prefix[doc_filename] == ref_doc_identifier and
                doc_index in ref_indices and
                doc_plant == ref_plant):
                match_found = True
                break
        # 2. RAG 결과 문서가 일반 작물 PDF 파일인 경우
        else:
            # 평가셋의 name이 RAG 결과 문서의 파일명(또는 작물명)과 같고,
            # 평가셋의 num(인덱스)이 RAG 결과 문서의 인덱스와 같고,
            # 평가셋의 p(작물명)가 RAG 결과 문서의 작물명(또는 파일명에서 추출한 작물명)과 같은지 확인
            # 여기서는 ref_doc_identifier가 작물명.pdf 형태이거나 작물명일 수 있음.
            # doc_filename은 작물명.pdf 형태. doc_plant는 작물명.
            
            # 비교 기준: 평가셋의 작물명(ref_plant)과 RAG 결과의 작물명(doc_plant)이 일치하고,
            # 평가셋의 문서 식별자(ref_doc_identifier, 예: 상추.PDF)와 RAG 결과 파일명(doc_filename)이 일치하고,
            # 평가셋의 인덱스(ref_indices)와 RAG 결과 인덱스(doc_index)가 일치하는지.
            if (ref_plant == doc_plant and
                ref_doc_identifier == doc_filename and # 또는 ref_doc_identifier가 작물명이고, doc_filename에서 작물명 추출 비교
                doc_index in ref_indices):
                match_found = True
                break
    
    y_true = [1] # 정답 문서는 존재한다고 가정
    y_pred = [1 if match_found else 0]

    # sklearn.metrics.f1_score 사용
    # zero_division=0: TP, FP, FN 모두 0일 경우 F1을 0으로 처리
    return f1_score(y_true, y_pred, zero_division=0)