# RAG/baseline/features/fact_checking.py
import streamlit as st
from typing import List, Tuple
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_upstage import UpstageGroundednessCheck
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, LANGSMITH_TRACING_ENABLED, LANGCHAIN_PROJECT, UPSTAGE_API_KEY, UPSTAGE_LABEL_TO_SCORE
from utils import sentence_split # 문장 분리 유틸리티

CALLBACKS_MULTITURN = []
if LANGSMITH_TRACING_ENABLED:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    TRACER_MULTITURN = LangChainTracer(project_name=f"{LANGCHAIN_PROJECT}-fact_checking")
    CALLBACKS_MULTITURN = [TRACER_MULTITURN]

# 단일 질문을 재작성. (Fact Checker 점수가 낮을 때 사용)
def rewrite_question_single(original_question: str, temperature: float = 0.3) -> str:
    if not OPENAI_API_KEY:
        st.error("질문 재작성을 위해서는 OpenAI API 키가 필요합니다.")
        return original_question

    prompt_template_text = """
    당신은 정보 검색 최적화를 위한 질문 리라이팅 도우미입니다.
    다음 질문을 더 나은 답변을 얻을 수 있도록, 더 명확하고 구체적인 형태로 바꿔주세요.
    원래 질문의 핵심 의도는 유지하되, 모호한 부분을 줄이세요.
    단, 원래 질문에 있던 작물 이름이나 핵심 키워드는 똑같이 유지해주세요.
    한 가지 형태로만 변환해 주세요.

    원래 질문: "{original_question}"
    재작성된 질문:
    """
    prompt = PromptTemplate.from_template(prompt_template_text)
    llm = ChatOpenAI(
        model="gpt-4o", # or gpt-3.5-turbo
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
        callbacks=CALLBACKS_MULTITURN
    )
    chain = prompt | llm

    try:
        response = chain.invoke({"original_question": original_question})
        rewritten = response.content.strip()
        if not rewritten: # LLM이 빈 문자열을 반환한 경우
            st.warning("질문 재작성 결과가 비어있습니다. 원본 질문을 사용합니다.")
            return original_question
        return rewritten
    except Exception as e:
        st.error(f"단일 질문 재작성 중 오류 발생: {e}")
        return original_question

# Upstage Fact checker function
def fact_checker(
    answer_sentences: List[str], # LLM 답변을 문장 단위로 분리한 리스트
    retrieved_docs: List[Document]
) -> Tuple[float, List[float]]:

    if not UPSTAGE_API_KEY:
        st.warning("Fact Checking을 위해서는 Upstage API 키가 필요합니다. 기본 점수 0.0을 반환합니다.")
        return 0.0, [0.0] * len(answer_sentences)

    if not answer_sentences:
        st.write("Fact Check 대상 문장이 없습니다.")
        return 0.0, []
    
    if not retrieved_docs:
        st.warning("Fact Check에 사용할 참고 문서가 없습니다. 기본 점수 0.0을 반환합니다.")
        return 0.0, [0.0] * len(answer_sentences)

    results_scores = []
    try:
        groundedness_check = UpstageGroundednessCheck(api_key=UPSTAGE_API_KEY)
    except Exception as e:
        st.error(f"Upstage GroundednessCheck 초기화 실패: {e}. Fact Checking을 건너뜁니다.")
        return 0.0, [0.0] * len(answer_sentences)

    st.write(f"답변 내 {len(answer_sentences)}개 문장에 대해 Fact Checking 수행 중... (Upstage API 비용 발생)")
    
    contexts_for_check = [doc.page_content for doc in retrieved_docs]
    # 모든 컨텍스트를 하나의 문자열로 합칠 수도 있지만, Upstage API의 컨텍스트 길이 제한 고려가 필요함
    # API 효율성을 위해 문맥을 합치는 것이 좋을 수 있으나, 일단 각 문서별로
    # 모든 문서를 합쳐서 하나의 긴 context로 만드는 것도 가능할듯?
    # combined_context = "\n\n".join(contexts_for_check)

    for i, sentence in enumerate(answer_sentences):
        if not sentence.strip(): # 빈 문장은 스킵
            results_scores.append(0.0)
            continue

        best_score_for_sentence = 0.0
        for doc_content in contexts_for_check:
            if not doc_content.strip(): continue # 빈 컨텍스트 스킵
            
            request_input = {"context": doc_content, "answer": sentence}
            try:
                # st.write(f"Checking sentence '{sentence[:30]}...' against doc part '{doc_content[:50]}...'") # 디버깅용
                res = groundedness_check.invoke(request_input) # API 호출
                score = UPSTAGE_LABEL_TO_SCORE.get(res, 0.0) # config에서 매핑된 점수 사용
                if score > best_score_for_sentence:
                    best_score_for_sentence = score
                if best_score_for_sentence == 1.0: # grounded면 더 볼 필요 없음
                    break 
            except Exception as e:
                st.warning(f"Upstage API 호출 중 오류 발생 (문장 {i+1}: '{sentence[:30]}...'): {e}")
                # 오류 발생 시 해당 문장은 0점으로 처리하고 다음 문장으로 넘어가거나, 해당 문서가 문제있다고 판단하고 다른 문서와 계속 비교 가능
                # 일단 해당 문서와의 비교 실패로 간주
        results_scores.append(best_score_for_sentence)

    if not results_scores:
        return 0.0, []
    
    average_score = sum(results_scores) / len(results_scores) if results_scores else 0.0
    return average_score, results_scores