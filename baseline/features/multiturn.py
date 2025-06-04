# RAG/baseline/features/multiturn.py
import streamlit as st
from langchain_openai import ChatOpenAI

from langchain.schema.messages import HumanMessage
from config import OPENAI_API_KEY, LANGSMITH_TRACING_ENABLED, LANGCHAIN_PROJECT

# LangSmith 설정
CALLBACKS_MULTITURN = []
if LANGSMITH_TRACING_ENABLED:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    TRACER_MULTITURN = LangChainTracer(project_name=f"{LANGCHAIN_PROJECT}-Multiturn")
    CALLBACKS_MULTITURN = [TRACER_MULTITURN]
    
# 이전 질문과 새 질문을 고려해 최종 사용할 질문을 LLM으로 재작성.
# Parameters:
# - query_db (list[str]): 이전 질문들. 가장 최근 질문은 마지막에 위치
# - question (str): 새로 들어온 사용자 질문
# - api_key (str): OpenAI API Key
# Returns:
# - str: 재작성된 질문 (또는 그대로 반환)
def query_reformulation(query_db: list[str], current_question: str, turn_num: int) -> str:
    if not OPENAI_API_KEY:
        st.error("멀티턴 질문 재구성을 위해서는 OpenAI API 키가 필요합니다.")
        return current_question
    
    # 첫 턴이거나, query_db가 비어있거나, 이전 질문이 없는 경우 (turn_num이 1인데 호출된 경우 등 escape)
    if turn_num <= 1 or not query_db or len(query_db) < turn_num -1 : # turn_num은 1부터 시작
        return current_question

    # query_db는 현재 질문이 추가된 후의 상태일 수 있으므로, 이전 질문은 turn_num-2 인덱스
    # (main_app.py에서 query_db.append 후 turn_num 증가, 그 후 이 함수 호출 가정)
    if turn_num - 2 < 0: # turn_num이 1일 때, query_db에 현재 질문만 들어간 경우
        previous_query = "" # 이전 질문 없음
    else:
        previous_query = query_db[turn_num-2] # 가장 최근의 이전 질문

    # 프롬프트 구성
    prompt_text = f"""
    당신은 사용자와의 다중 턴(multi-turn) 대화에서 문맥을 파악하여 현재 질문을 가장 효과적으로 재구성하는 AI 어시스턴트입니다.

    [이전 대화의 마지막 질문]:
    {previous_query if previous_query else "이전 대화 없음"}

    [사용자의 현재 새로운 질문]:
    {current_question}

    [지시사항]:
    1. 만약 현재 질문이 이전 질문에 이어지는 내용이라면, 이전 질문의 핵심 주제나 대상을 명시적으로 포함하여 현재 질문을 완전한 문장으로 재구성하세요.
       예시:
       - 이전: "배추의 주요 병충해는 무엇인가요?"
       - 현재: "특정 병원균은 어떤 것이 있나요?"
       - 재구성된 질문: "배추 병충해를 일으키는 특정 병원균에는 어떤 종류가 있나요?"

    2. 만약 현재 질문이 이전 질문과 독립적인 새로운 주제라면, 현재 질문을 그대로 사용하거나 문법적으로 더 자연스럽게 다듬으세요.
       예시:
       - 이전: "배추 재배 방법을 알려주세요."
       - 현재: "고구마가 물러요. 원인이 뭔가요?"
       - 재구성된 질문: "고구마가 무르는 원인이 무엇인가요?" (또는 "고구마가 물러요. 원인이 뭔가요?")

    3. 만약 현재 질문이 이전 질문에서 언급된 대상에 대한 단편적인 정보라면, 이전 질문의 대상과 현재 정보를 결합하여 완전한 질문으로 만드세요.
       예시:
       - 이전: "배추"
       - 현재: "노랗게 변해요."
       - 재구성된 질문: "배추 잎이 노랗게 변해요"

    4. 재구성된 질문은 명확하고 간결해야 합니다.
    5. 최종적으로 재구성된 질문 하나만 출력하세요.

    [재구성된 질문]:
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo", # 비용...
        temperature=0, # temperature 0 or 0.1?
        openai_api_key=OPENAI_API_KEY,
        max_tokens=256, # optional
        callbacks=CALLBACKS_MULTITURN
    )
    
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        reformulated_question = response.content.strip()
        if not reformulated_question: # LLM이 빈 문자열을 반환한 경우
            st.warning("멀티턴 질문 재구성 결과가 비어있습니다. 현재 질문을 그대로 사용합니다.")
            return current_question
        return reformulated_question
    except Exception as e:
        st.error(f"멀티턴 질문 재작성 중 오류 발생: {e}")
        return current_question # 오류 시 현재 질문 그대로 반환
    

###사람 질문과 AI 응답, 현재 history를 받아서 새로운 history를 생성하는 함수
from typing import Tuple

def summarize_conversation(
    current_history: str,
    current_question: str,
    current_AI_response: str
) -> Tuple[str, list]:
    if not OPENAI_API_KEY:
        st.error("멀티턴 질문 재구성을 위해서는 OpenAI API 키가 필요합니다.")
        return current_history, []
    
    # 프롬프트 구성
    prompt_text = f"""
당신은 대화 요약 및 히스토리 관리 전문가입니다.
다음 정보를 바탕으로:
1. 불필요한 부분은 제거하고 핵심만 간결하게 정리하여 **새로운 히스토리**를 생성하세요.
2. 요약된 히스토리를 대표하는 **주요 키워드 5개**를 추출하세요.

### 입력 정보
이전 히스토리:
{current_history}

사용자의 질문:
{current_question}

AI의 응답:
{current_AI_response}

### 출력 형식 (아래 포맷을 반드시 따르세요)
새로운 히스토리:
(요약된 히스토리)

대표 키워드:
- (키워드1)
- (키워드2)
- (키워드3)
- (키워드4)
- (키워드5)

### 요약 지침
- 대화 흐름을 이어갈 수 있도록 **중요한 키워드**, **의도**, **결과**만 포함하세요.
- **단계별 번호**나 **불릿** 없이, 자연스러운 **문장 형태**로 작성하세요.
- 히스토리 길이는 원래보다 **짧게 유지**하되, **문맥이 끊기지 않게** 작성하세요.

### 키워드 추출 지침
- 히스토리의 **핵심 단어 5개**를 선정하세요.
- **명확하고 구체적인 단어**로, **중복 없이** 추출하세요.
- **한국어**로 작성하고, **짧게** 표현하세요.
- 키워드만 추출하고, 추가 설명은 하지 마세요.
"""

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        max_tokens=512,
        callbacks=CALLBACKS_MULTITURN
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        output_text = response.content.strip()

        if "대표 키워드:" not in output_text:
            st.warning("LLM 응답 포맷이 올바르지 않습니다. 현재 History를 그대로 사용합니다.")
            return current_history, []
        
        # 결과 Parsing
        parts = output_text.split("대표 키워드:")
        new_history = parts[0].replace("새로운 히스토리:", "").strip()
        keywords_block = parts[1].strip()
        keywords = [kw.strip("- ").strip() for kw in keywords_block.splitlines() if kw.strip()]

        if not new_history or not keywords:
            st.warning("History 요약 또는 키워드 추출 결과가 비어 있습니다. 현재 History를 그대로 사용합니다.")
            return current_history, []
        
        return new_history, keywords

    except Exception as e:
        st.error(f"History 요약/키워드 추출 중 오류 발생: {e}")
        return current_history, []


    
    