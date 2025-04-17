# pip install streamlit python-dotenv langchain langchain-openai langchain-community sentence-transformers faiss-cpu langchain-upstage openai==1.* 
# Make sure openai version is >= 1.0
# -*- coding: utf-8 -*-
import os
import re
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import itertools # 모델 조합 생성용
import json # 평가셋 파일 로딩용
import numpy as np # 평균 계산용

# --- Langchain & AI Model Imports ---
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from sentence_transformers import CrossEncoder # For Reranking
from openai import OpenAI # 요약 & reranker & GPT Scoring
from langchain.prompts import PromptTemplate

#--- User Query Prompt ---
from query_prompt import build_user_query_prompt_chain, stream_final_answer_only

# --- langsmith  ---
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.schema.messages import HumanMessage

# --- upstage ----
from langchain_upstage import UpstageGroundednessCheck

# --- RAGAS ----
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy
)

from datasets import Dataset
import pandas as pd

# --- Score ----
from sklearn.metrics import f1_score


# --- Const and setting ---
BASE_DIR = Path(__file__).resolve().parent            # 사람마다 현재 폴더 설정 달라질 수 있으니 그에 따라 .parent 몇개 붙일지 결정정
FAISS_DB_PATH = Path(BASE_DIR / "faiss_db") # 위에 꺼 맞춰서 faiss_db 경로도 변경


# --- Load Env ---
load_dotenv(BASE_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
# Option for LangSmith tracing
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

# --- Langsmith Setting ---
PROJECT_NAME = "RAG-System"
TRACER = LangChainTracer(project_name=PROJECT_NAME)

# --- Model Definitions ---
AVAILABLE_EMBEDDINGS = {
    "bge-m3": "BAAI/bge-m3",
    "e5e": "intfloat/e5-large-v2",
    "openai": "text-embedding-3-large"
}
AVAILABLE_RERANKERS = ["ko-rerank", "mini", "bge"]
# --- Default Settings ---
DEFAULT_EMBEDDING_ALIAS = "bge-m3"
DEFAULT_RERANKER_METHOD = "ko-rerank"
DEFAULT_RETRIEVER_K = 8
DEFAULT_RERANKER_TOP_K = 4
DEFAULT_LLM_MODEL_NAME = "gpt-4o"
DEFAULT_GPT_SCORING_MODEL = "gpt-4o" # "gpt-4"

# Document Formatting
# document에서 내용만 뽑아서 str형태로 변환하는 함수 -> LLM 부를 때 이용
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


########################## RERANK functions ###############################
# GPT Document Summarization (Optional, used in summarize_and_rerank)
# GPT를 이용한 retrive 문서 512토큰 이하 요약 함수. API 비용 발생(Optional, gpt-3.5 쓰도록 고정)
# Use Langsmith for trace
def gpt_summarize_document(doc: Document, max_tokens=512, api_key=None) -> Document:
    if not api_key:
        raise ValueError("OpenAI API key is required for summarization.")
    # Use LangChain wrapper?
    llm = ChatOpenAI(model="gpt-4o", max_tokens=max_tokens, openai_api_key=api_key, callbacks=[TRACER])
    prompt = f"""다음 문서를 {max_tokens}토큰 이하로 요약해 주세요.\n\n문서:\n{doc.page_content}"""

    try:
        # Use LangChain method
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Store original text in metadata
        new_metadata = doc.metadata.copy()
        new_metadata["original_text"] = doc.page_content
        return Document(page_content=summary, metadata=new_metadata)
    except Exception as e:
        st.error(f"문서 요약 중 오류 발생: {e}")
        # Return original document if fail
        return doc

#입력 받은 reranker model 반환 함수
#rerank모델 받아오는 걸 한번만 하고 그 뒤엔 캐시에 저장된걸로 이용
@st.cache_resource(show_spinner=False) 
def get_reranker_model(method: str):
    method = method.lower()
    st.write(f"Reranker 모델 로딩 중: {method}...")
    if method == "ko-rerank":
        return CrossEncoder("Dongjin-kr/ko-reranker", device='cpu', max_length=512)
    elif method == "bge":
        return CrossEncoder("BAAI/bge-reranker-base", device='cpu', max_length=512)
    elif method == "mini":
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu', max_length=512)
    # gpt reranker 임시 제거
    else:
        raise ValueError(f"지원하지 않는 reranker method입니다: {method}")

#Reranker 모델 실행 함수
def rerank_documents(
    query: str,
    docs: List[Document],
    method: str,
    top_k: int = 3,
    openai_api_key: str = None # For future GPT reranker
) -> List[Tuple[Document, float]]:

    if method not in AVAILABLE_RERANKERS:
         st.warning(f"지원하지 않는 리랭커({method})입니다. 리랭킹을 건너<0xEB><0>뜁니다.")
         # Return original docs with dummy scores if method is invalid or None
         return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)][:top_k]

    try:
        reranker = get_reranker_model(method) # Load the specified model
        st.write(f"CrossEncoder ({method})를 사용하여 문서 재순위 평가 중...")
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        scored = list(zip(docs, scores))
        # Sort descending
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    except Exception as e:
        st.error(f"리랭킹 중 오류 발생 ({method}): {e}")
        # Fallback to returning top_k documents without reranking if error occurs
        return [(doc, 0.0) for doc in docs[:top_k]]

#요약 및 Rerank 함수
def summarize_and_rerank(
    query: str,
    docs: List[Document],
    summarize_first: bool = False, # Controls summarization
    method: str = "bge",
    top_k: int = 3,
    openai_api_key: str = None
) -> List[Document]:

    docs_to_rerank = docs
    if summarize_first:
        if not openai_api_key:
             st.warning("문서 요약을 위해서는 OpenAI API 키가 필요합니다. 요약을 건너뜁니다.")
        else:
            st.write("문서 요약 중 (Reranking 전)...")
            with st.spinner("요약 진행 중..."):
                docs_to_rerank = [
                    gpt_summarize_document(doc, max_tokens=512, api_key=openai_api_key) for doc in docs
                ]

    # Perform reranking
    reranked_with_scores = rerank_documents(
        query=query,
        docs=docs_to_rerank,
        method=method,
        top_k=top_k,
        openai_api_key=openai_api_key # For future
    )

    final_original_docs = []
    st.write("재순위화된 문서 (점수 순):")
    for i, (doc, score) in enumerate(reranked_with_scores):
        # Retrieve the original content from metadata if summarized, otherwise use page_content
        original_content = doc.metadata.get("original_text", doc.page_content)
        # Create a new Document object with the original content but keep metadata
        final_doc = Document(page_content=original_content, metadata=doc.metadata)
        final_original_docs.append(final_doc)

        # Display reranked doc info (optional)
        st.write(f"  - Rank {i+1} (Score: {score:.4f}): ", final_doc.metadata.get('file_name', 'N/A'), f" (Index: {final_doc.metadata.get('index', 'N/A')})")

    return final_original_docs



########  Vector DB function  #########

# Cache the loaded vector store
# 임베딩 모델 선택하고 모델에 따른 DB 불러오기 -> openai 임베딩 모델 불러오는건 추적 안됨
@st.cache_resource(show_spinner="벡터 DB 로딩 중...") #DB 받아오는 걸 한번만 하고 그 뒤엔 캐시에 저장된걸로 이용
def load_vector_store(faiss_alias, api_key):

    # Call embedding models
    ##### 임베딩 모델 불러오는 파트 #####
    if faiss_alias not in AVAILABLE_EMBEDDINGS:
        st.error(f"지원하지 않는 임베딩 모델 별칭입니다: {faiss_alias}")
        return None
    model_name = AVAILABLE_EMBEDDINGS[faiss_alias]

    st.write(f"임베딩 모델 로딩 중: {faiss_alias} ({model_name})...")
    embedding_model = None
    if faiss_alias == "openai":
        # OpenAI API 사용
        if not api_key:
            st.error("OpenAI 임베딩 모델을 로드하려면 OpenAI API 키가 필요합니다.")
            return None
        embedding_model = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
        st.write("(OpenAI 임베딩 사용 시 API 비용이 발생할 수 있습니다)")
    else:
        # 로컬 모델 사용
        try:
            # device = "cpu"
            device = "cpu" # device = "cuda" if torch.cuda.is_available()
            embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            st.error(f"HuggingFace 임베딩 모델 로딩 실패 ({model_name}): {e}")
            st.error("필요한 라이브러리가 설치되었는지 확인하세요: pip install sentence-transformers torch")
            return None

    ##### local에 저장된 벡터 DB 불러오는 파트 #####

    # full path to the FAISS index directory
    db_path = FAISS_DB_PATH / faiss_alias
    index_file_faiss = db_path / "index.faiss"
    index_file_pkl = db_path / "index.pkl"

    if not index_file_faiss.exists() or not index_file_pkl.exists():
        st.error(f"FAISS 인덱스 파일을 찾을 수 없습니다: {db_path}")
        st.error(f"먼저 '{faiss_alias}' 모델에 대한 벡터스토어를 생성하고 해당 경로에 저장했는지 확인하세요.")
        return None

    try:
        vectorstore = FAISS.load_local(
            folder_path=str(db_path), # Pass path as string
            embeddings=embedding_model,
            index_name="index", # Default index name used in saving
            allow_dangerous_deserialization=True # Needed for HuggingFaceEmbeddings
        )
        st.success(f"'{faiss_alias}' 벡터 DB 로드 완료!")
        return vectorstore
    except Exception as e:
        st.error(f"FAISS 벡터 DB 로드 중 오류 발생 ({faiss_alias}): {e}")
        return None




###########  Factchecker function  ##########

# Upstage Fact checker에 넘길 라벨
label_to_score={
    "grounded":1.0,
    "notSure":0.5,
    "notGrounded":0.0
}

# Rewrite User query -> Use "gpt-4o"
def rewrite_question_single(original_question: str, temperature: float = 0.3, api_key=None) -> str:
    if not api_key:
        st.error("질문 재작성을 위해서는 OpenAI API 키가 필요합니다.")
        return original_question # Return original if key is missing

    prompt_template = """
    당신은 정보 검색 최적화를 위한 질문 리라이팅 도우미입니다.
    다음 질문을 더 구체적이고 검색에 적합하게 바꿔주세요.
    단, 원래 질문에 있던 작물은 똑같이 유지해주세요.
    한 가지 형태로만 변환해 주세요.
    원래 질문: "{original_question}"
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4o", temperature=temperature, openai_api_key=api_key, callbacks=[TRACER])
    chain = prompt | llm

    try:
        response = chain.invoke({"original_question": original_question})
        return response.content
    except Exception as e:
        st.error(f"질문 재작성 중 오류 발생: {e}")
        return original_question # If error, return original

# 문서에서 원하는 내용 파싱하는 함수
def sentence_split(text):
    # If text is a string?
    if not isinstance(text, str):
        return []
    split_sentence=re.split(r'(?<=[-.!?])\s+',text)
    # Filter out empty strings that might result from splitting
    return [s for s in split_sentence if s]

# Upstage Fact checker function -> Use UpStage API 
def fact_checker(sentences: List[str], retrieved_docs: List[Document], api_key: str) -> Tuple[float, List[float]]:
    if not api_key:
        st.warning("Fact Checking을 위해서는 Upstage API 키가 필요합니다. 기본 점수 0.0을 반환합니다.")
        return 0.0, [0.0] * len(sentences)

    results = []
    try:
        groundedness_check = UpstageGroundednessCheck(api_key=api_key) # Pass API key
    except Exception as e:
        st.error(f"Upstage GroundednessCheck 초기화 실패: {e}. Fact Checking을 건너뜁니다.")
        return 0.0, [0.0] * len(sentences)

    st.write(f"문장 {len(sentences)}개에 대해 Fact Checking 수행 중... (Upstage API 비용 발생)")
    for i, sentence in enumerate(sentences):
        if not sentence:
            results.append(0.0)
            continue

        best_score_for_sentence = 0.0
        for doc in retrieved_docs:
            chunk = doc.page_content
            request_input = {"context": chunk, "answer": sentence}
            try:
                res = groundedness_check.invoke(request_input)
                score = label_to_score.get(res, 0.0)
                if score > best_score_for_sentence:
                    best_score_for_sentence = score
                    if best_score_for_sentence == 1.0: break
            except Exception as e:
                st.warning(f"Upstage API 호출 중 오류 발생 (문장: '{sentence[:30]}...'): {e}")
                best_score_for_sentence = 0.0
                break
        results.append(best_score_for_sentence)

    if not results: return 0.0, []
    average_score = sum(results) / len(results)
    return average_score, results



#################   GPT scoring  #################
#LLM + RAGAS 파싱 함수
def parse_combined_llm_scores(text: str) -> Dict[str, float]:
    pattern = r"(Intent Understanding|Semantic Similarity|Score Reliability):\s*([1-5])"
    matches = re.findall(pattern, text)
    return {label: int(score) / 5 for label, score in matches}

#LLM + RAGAS 스코어 평가 함수
def get_combined_score(query: str, response: str, context: str, ground_truth: str, api_key: str) -> Dict[str, Any]:
    # 1. RAGAS 평가
    ragas_dataset = Dataset.from_dict({
        "question": [query],
        "response": [response],
        "retrieved_contexts": [[context]],
        "reference": [ground_truth]
    })

    ragas_result = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy])
    ragas_df = ragas_result.to_pandas() 

    ragas_scores = {
        "faithfulness": round(ragas_df.loc[0, "faithfulness"], 3),
        "answer_relevancy": round(ragas_df.loc[0, "answer_relevancy"], 3)
    }

    # 2. LLM 평가 프롬프트
    prompt = f"""
[질문]
{query}

[Ground Truth]
{ground_truth}

[RAG 시스템의 응답]
{response}

당신은 농산물 RAG 시스템의 응답을 평가하는 전문가입니다. 아래 세 가지 항목에 대해 각각 1~5점 사이의 점수를 부여해주세요.  
가능한 한 **객관적**이고 일관된 기준에 따라 평가해 주세요.

---

### 1. Intent Understanding  
응답이 질문자의 **의도를 잘 이해했는지** 평가하세요.

- 5점: 질문에 정확히 답하고, 질문자 의도까지 파악하여 
    예시 : 병충해의 종류와 특징에 대해 설명하고, 방제법에 대해 설명하거나 사용자에게 추가 정보가 필요한지 물어봄.
- 3점: 질문에는 답했지만, 질문자의 의도를 파악하지 못함
    예시 : 병충해의 종류와 특징에 대해서 설명했지만, 어떻게 방제해야하는지 설명하지 않고 또한 사용자에게 추가정보가 필요한지 물어보지 않음.
- 1점: 질문을 오해하거나 질문의 내용과 무관한 응답, 또는 오탈자 포함  
    예시 : 병충해에 대해 물어보았는데 재배방법에 대해 응답답
또한, "답변할 수 없습니다"처럼 정직하게 회피한 경우도 높은 점수

---

### 2. Semantic Similarity  
Ground Truth와 응답이 **내용 및 의미 면에서 얼마나 유사한지** 평가하세요.

- 5점: 의미는 같고 표현만 다름
- 3점: 일부만 유사하거나 요약 수준
- 1점: 논지나 정보가 전혀 다름

---

### 3. Score Reliability  
당신이 지금 평가한 점수들이 **얼마나 신뢰할 수 있는지** 스스로 평가하세요.  
질문/문서/응답이 명확하여 평가가 쉬웠다면 높은 점수,  
정보가 부족하거나 애매했다면 낮은 점수.

---

### 출력 형식 (숫자만 아래 형식으로 출력하세요):

Intent Understanding: 4  
Semantic Similarity: 5  
Score Reliability: 5
"""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key, callbacks=[TRACER])
    
    llm_output = llm.invoke([HumanMessage(content=prompt)]).content
    llm_scores = parse_combined_llm_scores(llm_output)

    # 3. 합쳐서 반환
    return {
        **llm_scores,
        **ragas_scores
    }







##########################   LLM 답변 생성까지의 Pipeline function  ###############################

# 파이프라인 함수 : RAG 실행 및 LLM 답변 생성하는 것 까지 담당 -> streamlit UI에서 LLM 모델 선택하는 부분이 반영되는 곳
# OpenAPI API 사용
# Langsmith 추적
def run_rag_pipeline(
    question: str,
    vectorstore: FAISS,
    retriever_k: int,
    use_reranker: bool,
    reranker_method: Optional[str], # can None
    reranker_top_k: int,
    summarize_before_rerank: bool,
    llm_model_name: str,
    openai_api_key: str,
    run_id: str = ""
) -> Tuple[str, List[Document]]:
    """Run the Retrieve-Rerank(Optional)-Generate pipeline."""

    log_prefix = f"[{run_id}] " if run_id else ""

    # 1. Retrieve
    st.write(f"{log_prefix}1. 관련 문서 검색 중 (k={retriever_k})...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    initial_docs = retriever.invoke(question)
    st.write(f"{log_prefix}   초기 검색된 문서 {len(initial_docs)}개")

    # 2. Rerank (Optional)
    final_docs = []
    if use_reranker and reranker_method: # Check if method is provided
        st.write(f"{log_prefix}2. 문서 rerank 중 (Top {reranker_top_k}, Method: {reranker_method})...")
        final_docs = summarize_and_rerank(
             query=question,
             docs=initial_docs,
             summarize_first=summarize_before_rerank,
             method=reranker_method,
             top_k=reranker_top_k,
             openai_api_key=openai_api_key
         )
        if not final_docs:
             st.warning(f"{log_prefix}   rerank 후 유효한 문서가 없습니다. 초기 검색 결과 일부를 사용합니다.")
             final_docs = initial_docs[:reranker_top_k]
    elif use_reranker and not reranker_method:
         st.warning(f"{log_prefix}   reranker 사용이 선택되었으나 모델이 지정되지 않았습니다. reranking을 건너<0xEB>뜁니다.")
         final_docs = initial_docs[:reranker_top_k]
    else: # No reranking requested or no method specified
        st.write(f"{log_prefix}2. reranker 사용하지 않음. 초기 검색 결과 사용 (상위 {reranker_top_k}개).")
        final_docs = initial_docs[:reranker_top_k]

    if not final_docs:
         st.error(f"{log_prefix}   답변 생성에 사용할 문서를 얻지 못했습니다.")
         return "오류: 관련 문서를 찾거나 처리할 수 없습니다.", []


    # 3. Generate (LLM 호출)
    st.write(f"{log_prefix}3. AI 답변 생성 중 ({llm_model_name})... ")

    #################################################################
    # Few-shot-CoT 기반 프롬프트 체인 사용
    try:
        context = format_docs(final_docs)
        full_question = f"{question}\n\n참고 문서:\n{context}"

        cot_chain = build_user_query_prompt_chain()
        result = cot_chain.invoke({"question": full_question})

        # content 필드만 추출 (불필요한 메타데이터 제거)
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
        elif hasattr(result, "content"):
            content = result.content
        else:
            content = str(result)

        return content, final_docs

    except Exception as e:
        st.error(f"{log_prefix}   LLM 호출 중 오류 발생: {e}")
        return f"오류: 답변 생성 중 문제가 발생했습니다 ({e})", final_docs
    #################################################################

    # Define LLM
    llm = ChatOpenAI(model_name=llm_model_name, temperature=0, openai_api_key=openai_api_key, callbacks=[TRACER])

    # Define the RAG chain using the reranked context
    rag_chain = (
        # Pass the question and the pre-fetched/reranked context
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt_hub
        | llm
        | StrOutputParser()
    ).with_config({"callbacks": [TRACER]})# For callback tracing

    try:
        response = rag_chain.invoke(question)
        st.write(f"{log_prefix}   답변 생성 완료.")
        return response, final_docs # Return both response and the docs used for it
    except Exception as e:
        st.error(f"{log_prefix}   LLM 호출 중 오류 발생: {e}")
        return f"오류: 답변 생성 중 문제가 발생했습니다 ({e})", final_docs





############################## Evaluation Set Handling 함수 ################################
def load_evaluation_set(uploaded_file) -> Optional[List[Dict[str, Any]]]:
    if uploaded_file is None:
        st.error("평가셋 파일을 업로드해주세요.")
        return None

    try:
        uploaded_file.seek(0)
        eval_data = json.load(uploaded_file)

        required_keys = {'question', 'answer'}

        if not isinstance(eval_data, list):
            st.error("평가셋은 JSON 리스트 형식이어야 합니다.")
            return None

        valid_data = []

        for item in eval_data:
            if isinstance(item, dict) and required_keys.issubset(item):
                # name 또는 num이 비어 있을 경우 name을 "unrelate"로 설정
                is_name_empty = not item.get("name")
                is_num_empty = not item.get("num")

                if is_name_empty or is_num_empty:
                    item["name"] = ["unrelate"]  # 리스트 형태 유지

                valid_data.append(item)

        if len(valid_data) == 0:
            st.error("유효한 항목이 없습니다. 'question', 'answer', 'p', 'num', 'no', 'name' 키를 모두 포함해야 합니다.")
            return None

        st.success(f"평가셋 로드 완료: {len(valid_data)}개 질문")
        return valid_data

    except json.JSONDecodeError:
        st.error("JSON 파일 파싱 중 오류가 발생했습니다. 파일 형식을 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"평가셋 파일 처리 중 예외 발생: {e}")
        return None


############################## F1 Score 계산 함수 #############################

def compute_evalset_f1(
    evaluation_item: Dict[str, Any],
    used_docs: List[Document]
) -> float:
    """
    단일 평가 항목과 사용 문서 리스트 간 F1 Score 계산
    :param evaluation_item: 하나의 평가 항목 (딕셔너리)
    :param used_docs: LangChain Document 리스트 (metadata 기반)
    :return: F1 Score (float)
    """
    file_to_faq = {
        "filtered_plants1.json": "FAQ1",
        "filtered_plants2.json": "FAQ2"
    }

    # 평가 항목이 "unrelate"이면 제외
    if evaluation_item.get("name") == ["unrelate"]:
        return 0.0

    # 평가 항목 정보 추출
    ref_name = evaluation_item.get("name")[0] if isinstance(evaluation_item.get("name"), list) else evaluation_item.get("name")
    raw_nums = evaluation_item.get("num", [])
    ref_nums = [str(n) for n in raw_nums] if isinstance(raw_nums, list) else [str(raw_nums)]
    ref_plant = evaluation_item.get("p")[0] if isinstance(evaluation_item.get("p"), list) else evaluation_item.get("p")

    # 사용된 문서 metadata 기준 비교
    used_doc_metadata = [doc.metadata for doc in used_docs]

    match_found = False
    for meta in used_doc_metadata:
        file_name = meta.get("file_name", "")
        mapped_name = file_to_faq.get(file_name, file_name)
        index = str(meta.get("index"))
        plant = meta.get("plant")

        if mapped_name in ["FAQ1", "FAQ2"]:
            if (mapped_name == ref_name) and (index in ref_nums) and (plant == ref_plant):
                match_found = True
                break
        else:
            if (plant == ref_plant) and (index in ref_nums) and (mapped_name == ref_name):
                match_found = True
                break

    # 단일 항목 기준 정답 1, 예측이 맞으면 1 아니면 0
    y_true = [1]
    y_pred = [1 if match_found else 0]

    return f1_score(y_true, y_pred)




##########################  Streamli UI  #################################

# --- Streamlit App UI ---
st.set_page_config(page_title="RAG 시스템 (농산물 QA)", layout="wide")
st.title("RAG 기반 농산물 질의응답 시스템")

# --- Sidebar for SEtting ---
st.sidebar.title("RAG 설정")
st.sidebar.caption("비용 주의!")

# Model Selection for single run
st.sidebar.header("기본 실행 모델")
selected_embedding_alias = st.sidebar.selectbox(
    "임베딩 모델 선택",
    options=list(AVAILABLE_EMBEDDINGS.keys()),
    index=list(AVAILABLE_EMBEDDINGS.keys()).index(DEFAULT_EMBEDDING_ALIAS), # Default
    help="OpenAI 임베딩 모델 사용 시 API 비용이 발생."
)

# Reranker Selection for single run
use_reranker_default = st.sidebar.checkbox("리랭커 사용", value=True, help="답변 생성 전 검색된 문서의 순위를 재조정.")
selected_reranker_method_default = DEFAULT_RERANKER_METHOD # Default
if use_reranker_default:
    selected_reranker_method_default = st.sidebar.selectbox(
        "리랭커 모델 선택",
        options=AVAILABLE_RERANKERS,
        index=AVAILABLE_RERANKERS.index(DEFAULT_RERANKER_METHOD) # Default
    )
else:
     selected_reranker_method_default = None # No reranker if checkbox is off


# LLM Selection
selected_llm = st.sidebar.selectbox(
    "LLM 모델 선택",
    options=["gpt-3.5-turbo", "gpt-4", "gpt-4o"], #추후 추가
    index=0, # Default : gpt-3.5-turbo
    help="답변 생성에 사용될 언어 모델입니다. API 비용 발생."
)

# Parameter Tuning
st.sidebar.header("검색/재순위 파라미터")
retriever_k_value = st.sidebar.slider("초기 검색 문서 수 (Retriever K)", min_value=1, max_value=20, value=DEFAULT_RETRIEVER_K)
reranker_top_k_value = st.sidebar.slider("최종 문서 수 (Reranker Top K)", min_value=1, max_value=10, value=DEFAULT_RERANKER_TOP_K, help="리랭커 사용 시 최종적으로 LLM에게 전달될 문서의 수.")

# Optional Features Toggles
st.sidebar.header("부가 기능 (API 사용)")
summarize_before_rerank_toggle = st.sidebar.checkbox(
    "리랭킹 전 문서 요약 (OpenAI API)",
    value=False,
    help="리랭커가 처리하기 전에 각 문서를 OpenAI를 이용해 요약. API 비용이 발생."
)
use_fact_checker_toggle = st.sidebar.checkbox(
    "Fact Checker 사용 (Upstage API)",
    value=True,
    help="답변 생성 후 Upstage Groundedness Check를 수행. 점수가 낮으면 OpenAI를 이용해 1회 질문 재작성을 시도. Upstage API 비용이 발생."
)
use_gpt_scoring_toggle = st.sidebar.checkbox(
    f"GPT 자동 평가 (OpenAI API)({DEFAULT_GPT_SCORING_MODEL})",
    value=False,
    help=f"생성된 답변에 대해 ({DEFAULT_GPT_SCORING_MODEL})를 사용하여 관련성, 충실도 등을 평가. API 비용이 발생."
)




###############################     메인 코드     #################################

# --------- Main Area ---------
st.caption(f"현재 설정 | Embedding: {selected_embedding_alias} | Reranker: {'사용 안 함' if not use_reranker_default else selected_reranker_method_default} | LLM: {selected_llm}")
st.caption(f"부가 기능 | 요약: {'활성' if summarize_before_rerank_toggle else '비활성'} | FactCheck: {'활성' if use_fact_checker_toggle else '비활성'} | GPT 평가: {'활성' if use_gpt_scoring_toggle else '비활성'}")

# Load the primary vector store based on sidebar selection
# Cache the vector store based on the selected alias
@st.cache_resource(show_spinner="선택된 벡터 DB 로딩 중...")
def get_cached_vector_store(alias, key):
     # Wrap the original load function for caching with alias as key
     return load_vector_store(alias, key)

vectorstore = get_cached_vector_store(selected_embedding_alias, OPENAI_API_KEY)

if vectorstore:
    # Question Input
    question = st.text_input("질문을 입력하세요:", placeholder="예: 배추의 주요 병충해는 무엇인가요?")

    # --- Single Run Button ---
    if st.button("질문하기 (현재 설정 사용)") and question:
        st.markdown("---")
        st.header("단일 실행 결과")
        final_response = "오류: 처리 중 문제 발생"
        final_docs_used = []
        average_score = None # For fact checker

        try:
            # Indicate potential costs before running
            cost_flags = []
            if selected_embedding_alias == 'openai': cost_flags.append("OpenAI Embedding")
            if summarize_before_rerank_toggle: cost_flags.append("OpenAI Summarization")
            if selected_llm.startswith('gpt'): cost_flags.append(f"OpenAI LLM ({selected_llm})")
            if use_fact_checker_toggle: cost_flags.append("Upstage FactCheck")
            if use_gpt_scoring_toggle: cost_flags.append("OpenAI Scoring")
            if cost_flags:
                 st.info(f"API 비용 발생 가능: {', '.join(cost_flags)}")
            
            
            # 실험용 CoT 출력
            query_chain = build_user_query_prompt_chain()
            streamed = stream_final_answer_only(query_chain, question)

            st.markdown("### 🌱 프롬프트 기반 GPT 응답:")

            response_placeholder = st.empty()  # 빈 공간 하나 만들고
            response_text = ""
 
            for token in streamed:
                response_text += token
                response_placeholder.markdown(response_text)
            # ------------------------------------------------


            with st.spinner("RAG 파이프라인 실행 중..."):
                final_response, final_docs_used = run_rag_pipeline(
                    question=question,
                    vectorstore=vectorstore,
                    retriever_k=retriever_k_value,
                    use_reranker=use_reranker_default,
                    reranker_method=selected_reranker_method_default,
                    reranker_top_k=reranker_top_k_value,
                    summarize_before_rerank=summarize_before_rerank_toggle,
                    llm_model_name=selected_llm,
                    openai_api_key=OPENAI_API_KEY
                )

            # --- Optional: Fact Checking ---
            if use_fact_checker_toggle and final_docs_used:
                 with st.spinner("4. Fact Checking 중... (Upstage API)"):
                     sentences = sentence_split(final_response)
                     if sentences: # Only run if there are sentences
                         average_score, _ = fact_checker(sentences, final_docs_used, UPSTAGE_API_KEY)
                         st.info(f"Fact Check Score: {average_score:.4f}")

                         # Rewrite logic based on score (Uses OpenAI API)
                         if (average_score > 0.3) & (average_score < 0.7):
                             st.warning("Fact Check 점수가 낮아 질문을 재작성하여 다시 시도합니다. (OpenAI API 사용)")
                             rewritten_question = rewrite_question_single(question, api_key=OPENAI_API_KEY)
                             if rewritten_question != question: # Check if rewrite actually happened
                                 st.write(f"   재작성된 질문: {rewritten_question}")
                                 with st.spinner("재작성된 질문으로 RAG 파이프라인 재실행 중..."):
                                     # Re-run the pipeline with the rewritten question
                                     final_response, final_docs_used = run_rag_pipeline(
                                         question=rewritten_question, # Use rewritten question here
                                         vectorstore=vectorstore,
                                         retriever_k=retriever_k_value,
                                         use_reranker=use_reranker_default,
                                         reranker_method=selected_reranker_method_default,
                                         reranker_top_k=reranker_top_k_value,
                                         summarize_before_rerank=summarize_before_rerank_toggle,
                                         llm_model_name=selected_llm,
                                         openai_api_key=OPENAI_API_KEY,
                                         run_id="Rewrite" # Add identifier
                                     )
                                     # Optionally re-run fact-checker on the new response?
                             else:
                                st.warning("   질문 재작성에 실패했습니다. 원본 질문 결과를 사용합니다.")
                     else:
                          st.write("   Fact Check 점수가 임계치 범위 밖이거나 양호합니다.")
                     # Display final fact check score if calculated
                     if average_score is not None:
                          st.caption(f"Upstage Fact Check 평균 점수: {average_score:.4f}")

            # Display Final Results
            st.markdown("### AI 응답:")
            st.write(final_response)

            # --- Optional: GPT Scoring ---
            if use_gpt_scoring_toggle and final_docs_used:
                with st.spinner(f"5. GPT Scoring 중... ({DEFAULT_GPT_SCORING_MODEL})"):
                     #이거 이전에 F1 score 뽑아야 한다.
                     
                     context_str_for_scoring = format_docs(final_docs_used)
                     
                     combined_scores = get_combined_score(question, final_response, context_str_for_scoring,"예시시", OPENAI_API_KEY)
                     st.markdown("### GPT 평가 점수:")
                     st.json(combined_scores) # Display scores as JSON

            # Show context documents used for the final answer
            with st.expander("참고한 문서 (최종 답변 생성에 사용됨)"):
                if final_docs_used:
                    for i, doc in enumerate(final_docs_used):
                        st.markdown(f"**문서 {i+1}:**")
                        st.markdown(f"*출처: {doc.metadata.get('file_name', 'N/A')} (인덱스: {doc.metadata.get('index', 'N/A')})*")
                        st.text_area(label=f"문서 {i+1} 내용", value=doc.page_content, height=150, key=f"final_doc_{i}")
                else:
                    st.write("사용된 문서가 없습니다.")

        except Exception as e:
            st.error(f"단일 실행 처리 중 오류 발생: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show detailed traceback for debugging


    # --- Multi-Model Evaluation Section ---
    st.sidebar.markdown("---")
    st.sidebar.header("모델 조합 평가")
    st.sidebar.caption("선택된 모델들의 조합으로 RAG를 실행하고 결과를 비교합니다. 각 조합 실행 시 위 설정 및 부가 기능 옵션이 적용됩니다.")

    # Checkboxes for selecting models for evaluation run
    eval_embeddings = st.sidebar.multiselect(
        "평가할 임베딩 모델 선택",
        options=list(AVAILABLE_EMBEDDINGS.keys()),
        default=list(AVAILABLE_EMBEDDINGS.keys()), # Default to all
        help="평가에 사용할 임베딩 모델들을 선택합니다. 'openai' 선택 시 비용 발생."
    )

    eval_use_reranker = st.sidebar.checkbox("평가 시 reranker 사용", value=True)
    eval_rerankers = []
    if eval_use_reranker:
        eval_rerankers = st.sidebar.multiselect(
            "평가할 reranker 모델 선택",
            options=AVAILABLE_RERANKERS,
            default=AVAILABLE_RERANKERS # Default to all
        )

    # Evaluation Button
    if st.sidebar.button("모델 조합 평가 실행") and question:
        st.markdown("---")
        st.header("모델 조합 평가 결과")
        evaluation_results = []

        # Create combinations
        if not eval_embeddings:
             st.warning("평가를 위해 최소 하나 이상의 임베딩 모델을 선택해야 합니다.")
        elif eval_use_reranker and not eval_rerankers:
             st.warning("reranker 사용이 선택되었으나, 평가할 reranker 모델이 선택되지 않았습니다.")
        else:
            # Determine reranker list: either selected ones or [None] if not using reranker
            reranker_list_to_iterate = eval_rerankers if eval_use_reranker else [None]

            st.info(f"{len(eval_embeddings)}개 임베딩 모델과 {len(reranker_list_to_iterate)}개 리랭커 설정 조합으로 총 {len(eval_embeddings) * len(reranker_list_to_iterate)}회 평가를 실행합니다.")
            st.info(f"각 실행 시 부가 기능 설정이 적용됩니다.")

            # Loop through combinations
            for emb_alias in eval_embeddings:
                 # Load vector store for this embedding model (use caching)
                 current_vectorstore = get_cached_vector_store(emb_alias, OPENAI_API_KEY)
                 if not current_vectorstore:
                     st.error(f"평가 중단: '{emb_alias}' 벡터 DB 로드 실패.")
                     evaluation_results.append({
                         "Embedding": emb_alias, "Reranker": "N/A", "Response": "Vectorstore Load Failed",
                         "Fact Check Score": None, "Combined Score": None, "Used Docs": []
                     })
                     continue # Skip to next embedding model

                 for reranker_method in reranker_list_to_iterate:
                     reranker_display_name = '사용 안 함' if not eval_use_reranker or reranker_method is None else reranker_method
                     run_id = f"Emb: {emb_alias}, Rerank: {reranker_display_name}"
                     st.subheader(f"평가 실행 중: {run_id}")

                     try:
                         # Cost flags for this specific run
                         run_cost_flags = []
                         if emb_alias == 'openai': run_cost_flags.append("OpenAI Emb")
                         if summarize_before_rerank_toggle: run_cost_flags.append("OpenAI Sum")
                         if selected_llm.startswith('gpt'): run_cost_flags.append(f"OpenAI LLM")
                         if use_fact_checker_toggle: run_cost_flags.append("Upstage FC")
                         if use_gpt_scoring_toggle: run_cost_flags.append("OpenAI Score")
                         if run_cost_flags:
                             st.caption(f"API 비용 발생 가능: {', '.join(run_cost_flags)}")


                         with st.spinner(f"[{run_id}] RAG 파이프라인 실행 중..."):
                             # Use the core pipeline function
                             eval_response, eval_docs_used = run_rag_pipeline(
                                 question=question,
                                 vectorstore=current_vectorstore,
                                 retriever_k=retriever_k_value, # Use sidebar value
                                 use_reranker=eval_use_reranker, # Specific to eval run
                                 reranker_method=reranker_method, # Current reranker in loop
                                 reranker_top_k=reranker_top_k_value, # Use sidebar value
                                 summarize_before_rerank=summarize_before_rerank_toggle, # Use sidebar value
                                 llm_model_name=selected_llm, # Use sidebar value
                                 openai_api_key=OPENAI_API_KEY,
                                 run_id=run_id
                             )

                         # Optional Fact-Checking for each result (use main toggle)
                         avg_fact_check_score = None
                         if use_fact_checker_toggle and eval_docs_used:
                             with st.spinner(f"[{run_id}] Fact Checking 중... (Upstage API)"):
                                 sentences = sentence_split(eval_response)
                                 if sentences:
                                     avg_fact_check_score, _ = fact_checker(sentences, eval_docs_used, UPSTAGE_API_KEY)

                         # Optional GPT Scoring for each result (use main toggle)
                         combined_scores = None
                         if use_gpt_scoring_toggle and eval_docs_used:
                             with st.spinner(f"[{run_id}] GPT Scoring 중... (OpenAI API)"):
                                 context_str = format_docs(eval_docs_used)
                                 
                                 combined_scores = get_combined_score(question, eval_response, context_str,"예시시", OPENAI_API_KEY)


                         evaluation_results.append({
                             "Embedding": emb_alias,
                             "Reranker": reranker_display_name,
                             "Response": eval_response,
                             "Fact Check Score": avg_fact_check_score,
                             "Combined Score": combined_scores,
                             "Used Docs": eval_docs_used
                         })
                         st.success(f"[{run_id}] 완료.")

                     except Exception as e:
                         st.error(f"[{run_id}] 평가 실행 중 오류: {e}")
                         evaluation_results.append({
                             "Embedding": emb_alias,
                             "Reranker": reranker_display_name,
                             "Response": f"오류 발생: {e}",
                             "Fact Check Score": None,
                             "Combined Score": None,
                             "Used Docs": []
                         })

            # Display all eval results
            st.markdown("---")
            st.subheader("종합 평가 결과 요약")
            for i, result in enumerate(evaluation_results):
                 st.markdown(f"**{i+1}. Embedding: {result['Embedding']}, Reranker: {result['Reranker']}**")
                 st.markdown(f"**응답:**")
                 st.write(result['Response'])
                 if result['Fact Check Score'] is not None:
                     st.caption(f"Fact Check Score: {result['Fact Check Score']:.4f}")
                 if result['Combined Score'] is not None:
                     st.caption(f"Combined Score:")
                     st.json(result['Combined Score']) # Show GPT scores if available
                 with st.expander(f"사용된 문서 ({len(result['Used Docs'])}개)"):
                     if result['Used Docs']:
                         # Add index 'j' for the inner loop for unique keys
                         for j, doc in enumerate(result['Used Docs']):
                             st.markdown(f"**문서 {j+1}:**")
                             st.markdown(f"*출처: {doc.metadata.get('file_name', 'N/A')} (인덱스: {doc.metadata.get('index', 'N/A')})*")
                             st.text_area(
                                 label=f"문서 {j+1} 내용", # index
                                 value=doc.page_content,
                                 height=150,
                                 key=f"eval_doc_{i}_{j}" # unique key (결과 index i + 문서 index j)
                             )
                     else:
                         st.write("사용된 문서가 없습니다.")





    ######################## ---   평가셋 실험 코드   ---#####################
    st.sidebar.markdown("---")
    st.sidebar.header("모델 조합 평가 (평가셋 사용)")
    st.sidebar.caption("업로드된 평가셋 파일의 모든 질문을 사용하여 모델 조합별 성능을 평가하고 평균 점수를 계산합니다.")

    uploaded_eval_file = st.sidebar.file_uploader("평가셋 파일 업로드 (.json)", type=['json'], key="eval_file_uploader")

    eval_embeddings_set = st.sidebar.multiselect(
        "평가할 임베딩 모델 (평가셋)",
        options=list(AVAILABLE_EMBEDDINGS.keys()),
        default=list(AVAILABLE_EMBEDDINGS.keys()), # Default to all
        help="평가셋 평가에 사용할 임베딩 모델들을 선택."
    )

    eval_use_reranker_set = st.sidebar.checkbox("평가셋 평가 시 reranker 사용", value=True)
    eval_rerankers_set = []
    if eval_use_reranker_set:
        eval_rerankers_set = st.sidebar.multiselect(
            "평가할 reranker 모델 (평가셋)",
            options=AVAILABLE_RERANKERS,
            default=AVAILABLE_RERANKERS # Default to all
        )

    # Use session state to store results across potential reruns
    if 'evaluation_details' not in st.session_state:
        st.session_state.evaluation_details = {}
    if 'aggregated_scores' not in st.session_state:
        st.session_state.aggregated_scores = {}

    if st.sidebar.button("평가셋으로 모델 조합 평가 실행") and uploaded_eval_file and eval_embeddings_set:
        st.markdown("---")
        st.header("모델 조합 평가 결과 (평가셋)")

        evaluation_set = load_evaluation_set(uploaded_eval_file)

        if evaluation_set:
            # Reset previous results before starting a new evaluation
            st.session_state.evaluation_details = {}
            st.session_state.aggregated_scores = {}

            reranker_list_to_iterate_set = eval_rerankers_set if eval_use_reranker_set else [None]

            if eval_use_reranker_set and not eval_rerankers_set:
                st.warning("reranker 사용이 선택되었으나, 평가할 reranker 모델이 선택되지 않았습니다.")
            else:
                total_combinations = len(eval_embeddings_set) * len(reranker_list_to_iterate_set)
                total_questions = len(evaluation_set)
                st.info(f"{len(eval_embeddings_set)}개 임베딩, {len(reranker_list_to_iterate_set)}개 리랭커 설정으로 총 {total_combinations}회 조합 평가.")
                st.info(f"각 조합당 {total_questions}개 질문에 대해 평가를 실행합니다 (총 {total_combinations * total_questions}회 RAG 실행).")
                st.info(f"각 실행 시 부가 기능 설정이 적용됩니다.")

                progress_bar = st.progress(0.0)
                progress_text = st.empty()
                runs_completed = 0
                total_runs = total_combinations * total_questions

                for emb_alias in eval_embeddings_set:
                    current_vectorstore = get_cached_vector_store(emb_alias, OPENAI_API_KEY)
                    if not current_vectorstore:
                        st.error(f"평가 중단: '{emb_alias}' 벡터 DB 로드 실패.")
                        runs_completed += len(reranker_list_to_iterate_set) * total_questions
                        progress_bar.progress(min(1.0, runs_completed / total_runs) if total_runs > 0 else 0.0) # Avoid division by zero
                        continue

                    for reranker_method in reranker_list_to_iterate_set:
                        reranker_display_name = '사용 안 함' if not eval_use_reranker_set or reranker_method is None else reranker_method
                        combination_key = (emb_alias, reranker_display_name)

                        # Initialize lists for this combination in both dicts
                        if combination_key not in st.session_state.aggregated_scores:
                            st.session_state.aggregated_scores[combination_key] = {'fact_check_scores': [], 'combined_scores': [], 'f1_scores': []}
                        if combination_key not in st.session_state.evaluation_details:
                                st.session_state.evaluation_details[combination_key] = []

                        st.subheader(f"조합 평가 중: Emb: {emb_alias}, Rerank: {reranker_display_name}")

                        # Cost flags
                        run_cost_flags = []
                        if emb_alias == 'openai': run_cost_flags.append("OpenAI Emb")
                        if summarize_before_rerank_toggle: run_cost_flags.append("OpenAI Sum")
                        if selected_llm.startswith('gpt'): run_cost_flags.append(f"OpenAI LLM")
                        if use_fact_checker_toggle: run_cost_flags.append("Upstage FC")
                        if use_gpt_scoring_toggle: run_cost_flags.append(f"OpenAI Score (x{total_questions})")
                        if run_cost_flags: st.caption(f"API 비용 발생 가능: {', '.join(run_cost_flags)}")

                        for i, eval_item in enumerate(evaluation_set):
                            eval_question = eval_item['question']
                            eval_ground_truth = eval_item['answer']
                            run_id = f"Set_Emb:{emb_alias}_Rerank:{reranker_display_name}_Q:{i+1}"

                            progress_text.text(f"진행률: {runs_completed+1}/{total_runs} - {run_id}")

                            #response 및 score 선언언
                            eval_response = "오류"
                            eval_docs_used = []
                            avg_fact_check_score = None
                            avg_f1_score = None
                            combined_scores = None

                            try:
                                with st.spinner(f"[{run_id}] RAG 파이프라인 실행 중..."):
                                        eval_response, eval_docs_used = run_rag_pipeline(
                                            question=eval_question, vectorstore=current_vectorstore,
                                            retriever_k=retriever_k_value, use_reranker=eval_use_reranker_set,
                                            reranker_method=reranker_method, reranker_top_k=reranker_top_k_value,
                                            summarize_before_rerank=summarize_before_rerank_toggle,
                                            llm_model_name=selected_llm, openai_api_key=OPENAI_API_KEY,
                                            run_id=run_id
                                        )

                                if use_fact_checker_toggle and eval_docs_used:
                                        with st.spinner(f"[{run_id}] Fact Checking 중..."):
                                            sentences = sentence_split(eval_response)
                                            if sentences:
                                                avg_fact_check_score, _ = fact_checker(sentences, eval_docs_used, UPSTAGE_API_KEY)
                                        if avg_fact_check_score is not None:
                                            # Store for average calculation
                                            st.session_state.aggregated_scores[combination_key]['fact_check_scores'].append(avg_fact_check_score)

                                if use_gpt_scoring_toggle and eval_docs_used:
                                        with st.spinner(f"[{run_id}] Combined Scoring 중..."):
    #################################### Metadata 형식 ################################
    # eval_docs_used = 'plant': '상추', 'index': 87, 'file_name': '상추.PDF', 'character_num': 1390

    #평가셋의 형식만 알면 됨.
    #question', 'answer', 'p', 'num', 'no', 'name'를 key로 가짐
    #name과num이 없을 경우 "unrelate"를 저장하도록 함함

    #평가셋 청크는 
    #파일이름이 "filtered_plants1.json", "filtered_plants2.json"인 게 평가셋의 name의 FAQ1, FAQ2와 대응
    #PDF(작물.pdf 또는 작물.PDF로 저장)의 plant는 평가셋의의 name에 저장되어 있다.

    #eval_docs_used에 저장된 파일 이름이 FAQ1 or FAQ2가 아닐 경우, eval_docs_uised의 plant와 index, 그리고 평가셋 청크의 name과 num을 비교해서 다 일치하는게 있으면 True
    #eval_docs_used에 저장된 파일 이름이 FAQ1 or FAQ2일 경우, eval_docs_uised의 file_name과 index, plant와 그리고 평가셋 청크의 name과 num, p를 비교해서 다 일치하는게 있으면 True

    #이때 F1 score를 구하여라

    #그리고 name에 "unrelate"가 있으면 F1 score 계산에서 뺌.
                                            avg_f1_score = compute_evalset_f1(eval_item, eval_docs_used)
                                            #avg_f1_score = 1
                                            if avg_f1_score is not None:
                                                st.session_state.aggregated_scores[combination_key]['f1_scores'].append(avg_f1_score)
                                            
                                            context_str = format_docs(eval_docs_used)
                                            combined_scores = get_combined_score(query=eval_question, response=eval_response, context=context_str, ground_truth=eval_ground_truth, api_key=OPENAI_API_KEY)
                                        
                                        
                                        if combined_scores and "Error" not in combined_scores:
                                            # Store for average calculation
                                            st.session_state.aggregated_scores[combination_key]['combined_scores'].append(combined_scores)

                                # Store detailed results for this question regardless of errors in scoring
                                st.session_state.evaluation_details[combination_key].append({
                                        "question": eval_question,
                                        "ground_truth": eval_ground_truth,
                                        "response": eval_response,
                                        "fact_check_score": avg_fact_check_score, # May be None
                                        "combined_scores": combined_scores, # May be None or contain Error
                                        "used_docs": eval_docs_used
                                    })

                            except Exception as e:
                                st.error(f"[{run_id}] 평가 실행 중 오류: {e}")
                                # Store minimal detail on error
                                st.session_state.evaluation_details[combination_key].append({
                                    "question": eval_question,
                                    "ground_truth": eval_ground_truth,
                                    "response": f"오류 발생: {e}",
                                    "fact_check_score": None,
                                    "combined_scores": {"Error": str(e)},
                                    "used_docs": []
                                })
                            finally:
                                runs_completed += 1
                                progress_bar.progress(min(1.0, runs_completed / total_runs) if total_runs > 0 else 0.0)

                        progress_text.text(f"진행률: {runs_completed}/{total_runs} - 조합 완료: Emb: {emb_alias}, Rerank: {reranker_display_name}")

                progress_bar.progress(1.0)
                progress_text.text(f"평가 완료! 총 {runs_completed}/{total_runs} 실행 완료.")

                # --- Calculate and Display Average Scores ---
                st.markdown("---")
                st.subheader("평가셋 기반 모델 조합별 평균 점수")



                avg_results_display = []
                # Use st.session_state for accessing results
                for (emb_alias, reranker_name), scores_data in st.session_state.aggregated_scores.items():
                    num_successful_gpt_runs = len([s for s in scores_data.get('combined_scores', []) if s and "Error" not in s])
                    if num_successful_gpt_runs == 0 and not scores_data.get('fact_check_scores'):
                        avg_results_display.append(f"**Embedding: {emb_alias}, Reranker: {reranker_name}** (성공 질문 수: 0/{total_questions}) - 결과 없음")
                        avg_results_display.append("---")
                        continue

                    avg_fact_check = np.mean(scores_data['fact_check_scores']) if scores_data['fact_check_scores'] else None
                    avg_f1_score = np.mean(scores_data['f1_scores']) if scores_data['f1_scores'] else None

                    avg_combined_scores = {}

                    if scores_data['combined_scores']:
                        for s in scores_data['combined_scores']:
                            first_valid_score = next((s for s in scores_data['combined_scores'] if s and "Error" not in s), None)
                        
                        if first_valid_score:
                                score_keys = first_valid_score.keys()
                                for key in score_keys:
                                    key_scores = [s[key] for s in scores_data['combined_scores'] if s and "Error" not in s and key in s and s[key] is not None]
                                    if key_scores:
                                        avg_combined_scores[f"Avg. {key}"] = np.mean(key_scores)
                                    else:
                                        avg_combined_scores[f"Avg. {key}"] = None

                    result_line = f"**Embedding: {emb_alias}, Reranker: {reranker_name}** (성공 질문 수: {num_successful_gpt_runs}/{total_questions})"
                    avg_results_display.append(result_line)

                    if avg_fact_check is not None: avg_results_display.append(f"  - Avg. Fact Check Score: {avg_fact_check:.4f}")
                    if avg_f1_score is not None: avg_results_display.append(f"  - Avg. F1 Score : {avg_f1_score:.4f}")
                    if avg_combined_scores:
                            for key, avg_val in avg_combined_scores.items():
                                avg_results_display.append(f"  - {key}: {avg_val:.4f}" if avg_val is not None else f"  - {key}: N/A")
                    
                    
                    avg_results_display.append("---")

                st.markdown("\n".join(avg_results_display))

                # --- Display Detailed Results ---
                st.markdown("---")
                st.subheader("평가셋 기반 모델 조합별 상세 결과")

                # Use st.session_state for accessing details
                if not st.session_state.evaluation_details:
                        st.write("상세 결과가 없습니다.")
                else:
                        for (emb_alias, reranker_name), details_list in st.session_state.evaluation_details.items():
                            expander_label = f"Embedding: {emb_alias}, Reranker: {reranker_name} (총 {len(details_list)}개 질문)"
                            with st.expander(expander_label):
                                if not details_list:
                                    st.write("이 조합에 대한 결과가 없습니다.")
                                    continue

                                for i, item in enumerate(details_list):
                                    st.markdown(f"**질문 {i+1}:** {item['question']}")
                                    st.markdown(f"**정답:** {item['ground_truth']}")
                                    st.markdown(f"**AI 응답:**")
                                    st.write(item['response']) # Use st.write for potentially long answers

                                    # Display scores if available
                                    if item['fact_check_score'] is not None:
                                        st.caption(f"Fact Check Score: {item['fact_check_score']:.4f}")
                                    
                                    if item['combined_scores'] is not None:
                                        if "Error" in item['combined_scores']:
                                            st.error(f"Combined Score Error: {item['combined_scores']['Error']}")
                                        else:
                                            st.caption("Combined Score:")
                                            st.json(item['combined_scores'])

                                    # Display
                                    used_docs = item.get('used_docs', [])
                                    st.markdown("---")
                                    if used_docs:
                                        st.markdown(f"**참고한 문서 ({len(used_docs)}개):**")
                                        for j, doc in enumerate(used_docs):
                        
                                            doc_metadata = getattr(doc, 'metadata', {})
                                            doc_page_content = getattr(doc, 'page_content', 'N/A')

                                            st.markdown(f"**문서 {j+1}:**")
                                            st.markdown(f"*출처: {doc_metadata.get('file_name', 'N/A')} (인덱스: {doc_metadata.get('index', 'N/A')})*")

                                            # Unique key for the text_area inside
                                            doc_content_key = f"detail_doc_content_{emb_alias}_{reranker_name}_{i}_{j}"

                                            st.text_area(
                                                label=f"문서 {j+1} 내용",
                                                value=doc_page_content,
                                                height=100,
                                                key=doc_content_key
                                            )
                                        st.markdown("<br>", unsafe_allow_html=True) # 문서 목록 아래에 약간의 공간 추가
                                    else:
                                        st.markdown("**참고한 문서:** 없음") # 문서가 없을 때 표시
                                    
                                    # st.markdown("---") # Separator between questions

# Display results even if the button wasn't clicked in this run (results might be in session state)
elif st.session_state.evaluation_details:
    st.markdown("---")
    st.subheader("이전 평가 결과 (평가셋 기반)")
    st.info("새로운 평가를 실행하려면 평가셋 파일을 업로드하고 '평가셋으로 모델 조합 평가 실행' 버튼을 클릭하세요.")

else:
    st.warning("벡터 DB를 로드할 수 없습니다. 사이드바에서 선택된 임베딩 모델에 대한 FAISS 인덱스가 '{FAISS_DB_PATH}' 경로에 있는지 확인해주세요.")

st.markdown("---")
st.caption("Powered by LangChain, OpenAI, FAISS, HuggingFace, Upstage & Streamlit")