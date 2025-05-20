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

#        ### 1. Intent Understanding
#        응답이 질문자의 **의도를 잘 이해했는지** 평가하세요.   
#     - 5점: 질문에 정확히 답하고, 질문자 의도까지 파악하여 
#     예시 : 병충해의 종류와 특징에 대해 설명하고, 방제법에 대해 설명하거나 사용자에게 추가 정보가 필요한지 물어봄.
# - 3점: 질문에는 답했지만, 질문자의 의도를 파악하지 못함
#     예시 : 병충해의 종류와 특징에 대해서 설명했지만, 어떻게 방제해야하는지 설명하지 않고 또한 사용자에게 추가정보가 필요한지 물어보지 않음.
# - 1점: 질문을 오해하거나 질문의 내용과 무관한 응답, 또는 오탈자 포함  
#     예시 : 병충해에 대해 물어보았는데 재배방법에 대해 응답답
# 또한, "답변할 수 없습니다"처럼 정직하게 회피한 경우도 높은 점수
    llm_eval_scores = {}
    try:
        st.write(f"LLM ({DEFAULT_GPT_SCORING_MODEL}) 기반 추가 평가 중...")
        prompt_text = f"""
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
        - 5점: 질문에 정확히 답하고, 질문자 의도까지 파악하여 필요한 추가 정보를 제공하거나 제안함.
        - 3점: 질문에는 답했지만, 질문자의 핵심 의도를 완전히 파악하지 못했거나 부가 정보가 부족함.
        - 1점: 질문을 오해하거나 질문의 내용과 무관한 응답, 또는 심각한 오탈자 포함.
        (참고: "답변할 수 없습니다"처럼 정직하게 회피한 경우, 질문의 의도를 이해하고 적절히 대응한 것으로 간주하여 3~4점 부여 가능)

        ---

        ### 2. Semantic Similarity (vs Ground Truth)
        Ground Truth와 응답이 **내용 및 의미 면에서 얼마나 유사한지** 평가하세요.
        - 5점: 의미는 같고 표현만 다름
        - 3점: 일부만 유사하거나 요약 수준
        - 1점: 논지나 정보가 전혀 다름

        ---

        ### 3. Score Reliability
        당신이 지금 평가한 점수들이 **얼마나 신뢰할 수 있는지** 스스로 평가하세요.
        질문, Ground Truth, RAG 응답이 명확하여 평가가 쉬웠다면 높은 점수.
        정보가 부족하거나 애매모호하여 평가에 어려웠다면 낮은 점수.

        ---

        ### 출력 형식 (숫자만 아래 형식으로 출력하세요):
        Intent Understanding: 4  
        Semantic Similarity: 5  
        Score Reliability: 5
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