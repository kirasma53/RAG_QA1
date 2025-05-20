# RAG/baseline/features/summarization.py
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.schema.messages import HumanMessage
from config import OPENAI_API_KEY, LANGSMITH_TRACING_ENABLED, LANGCHAIN_PROJECT, DEFAULT_GPT_SUMMARIZATION_MODEL

# LangSmith 설정
CALLBACKS = []
if LANGSMITH_TRACING_ENABLED:
    from langchain.callbacks.tracers.langchain import LangChainTracer
    TRACER_SUMMARIZE = LangChainTracer(project_name=f"{LANGCHAIN_PROJECT}-Summarization")
    CALLBACKS = [TRACER_SUMMARIZE]

# gpt 요약
def gpt_summarize_document(doc: Document, max_tokens=512) -> Document:
    if not OPENAI_API_KEY:
        st.warning("문서 요약을 위해서는 OpenAI API 키가 필요합니다. 요약을 건너뜁니다.")
        return doc

    llm = ChatOpenAI(
        model=DEFAULT_GPT_SUMMARIZATION_MODEL,
        max_tokens=max_tokens,
        openai_api_key=OPENAI_API_KEY,
        callbacks=CALLBACKS
    )
    prompt_text = f"""다음 문서를 한국어로 {max_tokens}토큰 이하로 요약해 주세요.\n\n문서:\n{doc.page_content}"""

    try:
        # Use LangChain method
        response = llm.invoke([HumanMessage(content=prompt_text)])
        summary = response.content.strip()

        # Store original text in metadata
        new_metadata = doc.metadata.copy()
        new_metadata["original_text"] = doc.page_content
        return Document(page_content=summary, metadata=new_metadata)
    except Exception as e:
        st.error(f"문서 요약 중 오류 발생 (문서 ID: {doc.metadata.get('id', 'N/A')}): {e}")
        # Return original document if fail
        return doc