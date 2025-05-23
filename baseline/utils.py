# RAG/baseline/utils.py
import re
from typing import List
from langchain_core.documents import Document

# document에서 내용만 뽑아서 str형태로 변환하는 함수 -> LLM 부를 때 이용
def format_docs(docs: List[Document]) -> str:
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    
# 텍스트를 문장 단위로 분리
def sentence_split(text: str) -> List[str]:
    # If text is a string?
    if not isinstance(text, str):
        return []
    # 문장 부호 뒤에 공백이 여러 개 있거나 없는 경우도 고려
    split_sentence = re.split(r'(?<=([.!?]))\s+', text.strip())
    # 빈 문자열 제거 및 앞뒤 공백 제거
    return [s.strip() for s in split_sentence if s and s.strip()]