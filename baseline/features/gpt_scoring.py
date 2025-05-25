# RAG/baseline/features/gpt_scoring.py
import re
import streamlit as st
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from ragas import evaluate as ragas_evaluate # ragas.evaluate와 streamlit.evaluate 이름 충돌 방지
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

from config import OPENAI_API_KEY, LANGSMITH_TRACING_ENABLED, LANGCHAIN_PROJECT, DEFAULT_GPT_SCORING_MODEL

# LangSmith 설정
CALLBACKS = []
if LANGSMITH_TRACING_ENABLED:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    TRACER_SCORING = LangChainTracer(project_name=f"{LANGCHAIN_PROJECT}-GPTScoring")
    CALLBACKS = [TRACER_SCORING]

# LLM + RAGAS 파싱 함수
# LLM 기반 평가 결과 텍스트를 파싱 -> return score dictionary
def parse_combined_llm_scores(text: str) -> Dict[str, float]:
    scores = {}
    #가독성
    patterns = {
        "Intent Understanding": r"Intent Understanding:\s*([1-5])",
        "Semantic Similarity": r"Semantic Similarity:\s*([1-5])",
        "Score Reliability": r"Score Reliability:\s*([1-5])"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            # 점수를 0.0 ~ 1.0 사이로 정규화 (5점 만점 기준)
            scores[key] = int(match.group(1)) / 5.0
        else:
            scores[key] = 0.0 # 매칭되는 항목이 없으면 0점 처리 또는 None
    return scores

# RAGAS + LLM 하여 점수 반환 
def get_combined_score(
    query: str,
    response: str,
    context: str, # 검색된 문서들의 내용을 합친 문자열
    ground_truth: str
) -> Dict[str, Any]:

    if not OPENAI_API_KEY:
        st.error("GPT 자동 평가를 위해서는 OpenAI API 키가 필요합니다.")
        return {"Error": "OpenAI API Key not found"}

    # 1. RAGAS 평가 (Faithfulness, Answer Relevancy)
    ragas_scores = {}
    try:
        st.write("RAGAS 평가 중 (Faithfulness, Answer Relevancy)...")
        # RAGAS는 컨텍스트를 리스트 형태로 받음
        retrieved_contexts_list = [context] if isinstance(context, str) else context
        if not isinstance(retrieved_contexts_list, list): # 혹시 다른 타입이면 리스트로
             retrieved_contexts_list = [str(retrieved_contexts_list)]

        # response가 비어있거나 context가 비어있으면 RAGAS 평가 불가
        if not response or not retrieved_contexts_list or not retrieved_contexts_list[0]:
            st.warning("RAGAS 평가를 위한 응답 또는 컨텍스트가 충분하지 않습니다.")
            ragas_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        else:
            dataset = Dataset.from_dict({
                "question": [query],
                "answer": [response],
                "retrieved_contexts": [retrieved_contexts_list],
                "reference": [ground_truth]
            })
            
            # RAGAS 평가 실행
            ragas_result = ragas_evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
                # llm=ChatOpenAI(model_name=DEFAULT_GPT_SCORING_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0) # RAGAS 내부 LLM 지정 가능
                # raise_exceptions=False # 오류 발생 시 평가 중단 방지
            )
            ragas_df = ragas_result.to_pandas()
            
            ragas_scores = {
                "faithfulness": round(ragas_df.loc[0, "faithfulness"], 3) if "faithfulness" in ragas_df.columns else 0.0,
                "answer_relevancy": round(ragas_df.loc[0, "answer_relevancy"], 3) if "answer_relevancy" in ragas_df.columns else 0.0
            }
    except Exception as e:
        st.error(f"RAGAS 평가 중 오류 발생: {e}")
        ragas_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0, "RAGAS_Error": str(e)}


    # 2. LLM 평가 프롬프트 (Intent Understanding, Semantic Similarity, Score Reliability)

        # ### 1. Intent Understanding
        # 응답이 질문자의 **의도를 잘 이해했는지** 평가하세요.
        # - 5점: 질문에 정확히 답하고, 질문자 의도까지 파악하여 필요한 추가 정보를 제공하거나 제안함.
        # - 3점: 질문에는 답했지만, 질문자의 핵심 의도를 완전히 파악하지 못했거나 부가 정보가 부족함.
        # - 1점: 질문을 오해하거나 질문의 내용과 무관한 응답, 또는 심각한 오탈자 포함.
        # (참고: "답변할 수 없습니다"처럼 정직하게 회피한 경우, 질문의 의도를 이해하고 적절히 대응한 것으로 간주하여 3~4점 부여 가능)
    llm_eval_scores = {}
    try:
        st.write(f"LLM ({DEFAULT_GPT_SCORING_MODEL}) 기반 추가 평가 중...")
        prompt_text = f"""
당신은 **농산물 QA 시스템의 평가자 역할**을 맡은 LLM입니다. 당신의 과제는 아래에 제시된 질문, 정답(Ground Truth), 그리고 시스템 응답(RAG 응답)을 바탕으로, **모델이 얼마나 잘 응답했는지**를 평가하는 것입니다.  
각 평가 항목은 **1~5점 사이의 정수 점수**로 채점해야 하며, 점수 기준은 아래에 구체적으로 설명되어 있습니다.

---

## 배경 정보
- 이 시스템은 농산물 관련 질문에 대해 정확하고 맥락에 맞는 답변을 생성하는 Retrieval-Augmented Generation (RAG) 모델입니다.
- 평가자는 LLM이며, 사람이 납득할 수 있는 신뢰도 있는 채점을 자동으로 수행하는 것이 목적입니다.
- 아래 세 항목은 각각 독립적으로 평가됩니다.

---

## 평가 항목 1: Intent Understanding (질문 의도 파악)
**정의:** 모델이 질문자의 의도를 정확히 이해하고, 핵심적인 답변과 함께 적절한 추가 정보 또는 제안을 제공했는지를 평가하세요.

### 점수 기준:
- **5점:** 질문에 명확히 답하고, 의도를 파악하여 유의미한 제안 또는 추가 설명까지 포함함  
  - 예시: “잎이 누렇게 마릅니다” → 병해 원인 설명 + 방제법 제안 + 추가 증상 확인 질문
- **3점:** 질문에는 대체로 응답했지만, 사용자의 목적을 충분히 반영하지 못함  
  - 예시: 병해 원인만 설명하고 방제법은 누락함
- **1점:** 질문을 오해했거나 무관한 내용을 응답함 / 명백한 오류 또는 심각한 오탈자 포함  
  - 예시: 병해 질문에 생육 환경만 설명

**긍정적 제약 조건:** “답변할 수 없습니다”와 같이 정직한 회피 응답은 질문 의도를 이해한 대응으로 간주할 수 있으며, **3~4점** 부여 가능합니다.

---

## 평가 항목 2: Semantic Similarity (vs Ground Truth)
**정의:** 응답이 Ground Truth와 의미적으로 얼마나 가까운지를 판단하세요. 단순 표현 차이는 감점 요인이 아닙니다.

### 점수 기준:
- **5점:** 정보와 의미가 모두 동일하며, 표현만 다름  
  - 예시: 방제법은 유사한 설명이나 순서나 문장이 다름
- **3점:** 일부 정보만 포함되었거나 요약 수준의 응답  
  - 예시: 병해 이름만 같고 방제법은 빠짐
- **1점:** 주된 정보가 다르거나 사실과 반대됨  
  - 예시: 다른 병해에 대한 설명, 잘못된 방제법

**예외 상황:** Ground Truth가 매우 구체적인 경우, 개괄적 요약이더라도 **핵심 의미를 담고 있다면** 3~4점 가능.

---

## 평가 항목 3: Score Reliability (신뢰도 평가)
**정의:** 당신이 위에서 매긴 점수들이 **얼마나 객관적이고 일관되게 평가될 수 있었는지** 판단하세요.

### 점수 기준:
- **5점:** 질문, Ground Truth, 응답이 모두 명확하여 평가가 쉬웠음 (재평가 시에도 점수가 같을 것 같음)
- **3점:** 일부 애매한 표현 또는 정보 부족으로 인해 점수에 약간의 불확실성 존재
- **1점:** 질문이나 응답이 모호하거나 정보가 부족하여 판단이 매우 어려움

**긍정적 제약 조건:** 판단이 애매하더라도 기준에 최대한 맞춰 일관되게 채점하도록 노력하세요.

---

## 출력 형식
아래 형식처럼 **숫자만 출력**하세요 (설명은 필요 없음):

Intent Understanding: 4  
Semantic Similarity: 5  
Score Reliability: 5

---

[질문]  
{query}

[Ground Truth]  
{ground_truth}

[RAG 시스템의 응답]  
{response}
"""

        llm = ChatOpenAI(
            model_name=DEFAULT_GPT_SCORING_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            callbacks=CALLBACKS
        )
        
        llm_output = llm.invoke([HumanMessage(content=prompt_text)]).content
        llm_eval_scores = parse_combined_llm_scores(llm_output)
    except Exception as e:
        st.error(f"LLM 기반 추가 평가 중 오류 발생: {e}")
        llm_eval_scores = {"LLM_Eval_Error": str(e)}

    # 3. 모든 점수 합쳐서 반환
    combined_scores_result = {**ragas_scores, **llm_eval_scores}
    return combined_scores_result