# RAG/baseline/main_app.py
# pip install streamlit python-dotenv langchain langchain-openai langchain-community sentence-transformers faiss-cpu langchain-upstage openai==1.*
# Make sure python version is = 3.12.*

# -*- coding: utf-8 -*-
import streamlit as st
import itertools # 현재는 사용 X
import json # 현재는 사용 X
import numpy as np # 평균 점수 계산 시 사용
import pandas as pd # 현재는 사용 X

import os
from pathlib import Path # config에서 사용

# --- Import config, utils, rag_core ---
from config import (
    OPENAI_API_KEY, UPSTAGE_API_KEY, FAISS_DB_PATH, # API 키는 각 모듈에서 직접 config 참조하지만 혹시 모르니까
    AVAILABLE_EMBEDDINGS, AVAILABLE_RERANKERS,
    DEFAULT_EMBEDDING_ALIAS, DEFAULT_RERANKER_METHOD,
    DEFAULT_RETRIEVER_K, DEFAULT_RERANKER_TOP_K,
    DEFAULT_LLM_MODEL_NAME, DEFAULT_GPT_SCORING_MODEL,
    LANGSMITH_TRACING_ENABLED # LangSmith 사용 여부 표시용
)
from utils import format_docs, sentence_split
from rag_core import (
    load_vector_store, run_rag_pipeline,
    load_evaluation_set, compute_evalset_f1
)
# --- Import feature module ---
from features.multiturn import query_reformulation, summarize_conversation
from features.fact_checking import fact_checker, rewrite_question_single
from features.gpt_scoring import get_combined_score



# --- Streamlit App UI ---
st.set_page_config(page_title="RAG 시스템 (농산물 QA)", layout="wide")
st.title("RAG 기반 농산물 질의응답 시스템")

# --- Sidebar for Settings ---
st.sidebar.title("RAG 시스템 설정")
st.sidebar.caption(f"LangSmith Tracing: {'✅ Enabled' if LANGSMITH_TRACING_ENABLED else '❌ Disabled'}") ## 이모지 넣는게 이쁠듯
st.sidebar.markdown("---")

# --- 모델 및 파라미터 선택 ---
st.sidebar.header("1. 기본 실행 모델 및 파라미터")
selected_embedding_alias = st.sidebar.selectbox(
    "임베딩 모델 선택", list(AVAILABLE_EMBEDDINGS.keys()),
    index=list(AVAILABLE_EMBEDDINGS.keys()).index(DEFAULT_EMBEDDING_ALIAS),
    help="문서와 질문을 벡터로 변환하는 모델입니다. 'openai' 선택 시 비용 발생."
)
selected_llm = st.sidebar.selectbox(
    "LLM 모델 선택", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
    index=["gpt-3.5-turbo", "gpt-4", "gpt-4o"].index(DEFAULT_LLM_MODEL_NAME),
    help="답변 생성에 사용될 언어 모델입니다. API 비용 발생."
)
use_reranker_default = st.sidebar.checkbox(
    "리랭커 사용", value = False,
    help="검색된 문서의 순위를 답변 생성 전에 재조정합니다."
)
selected_reranker_method_default = None
if use_reranker_default:
    selected_reranker_method_default = st.sidebar.selectbox(
        "리랭커 모델 선택", AVAILABLE_RERANKERS,
        index=AVAILABLE_RERANKERS.index(DEFAULT_RERANKER_METHOD) if DEFAULT_RERANKER_METHOD in AVAILABLE_RERANKERS else 0,
        help="문서 재순위화에 사용될 모델입니다."
    )
retriever_k_value = st.sidebar.slider(
    "초기 검색 문서 수 (Retriever K)", min_value=1, max_value=20, value=DEFAULT_RETRIEVER_K,
    help="벡터 DB에서 질문과 유사한 문서를 검색할 최대 개수입니다."
)
reranker_top_k_value = st.sidebar.slider(
    "최종 문서 수 (Reranker Top K)", min_value=1, max_value=10, value=DEFAULT_RERANKER_TOP_K,
    help="리랭커 사용 시, LLM에게 최종적으로 전달될 문서의 수입니다."
)
st.sidebar.markdown("---")

# --- 부가 기능 On/Off ---
st.sidebar.header("2. 부가 기능 On/Off")
use_query_multiturn_toggle = st.sidebar.checkbox(
    "query 요약 멀티턴 사용", value= False,
    help="이전 질문을 기반으로 현재 질문을 재구성합니다. (OpenAI API 비용 발생 가능)"
)
use_history_multiturn_toggle = st.sidebar.checkbox(
    "history 요약 멀티턴 사용", value = False,
    help = "이전 대화 내용의 요약을 기반으로 현재 질문을 재구성합니다. (OpenAI API 비용 발생가능)"
)
use_cot_toggle = st.sidebar.checkbox(
    "CoT 프롬프트 사용", value=False,
    help="LLM이 단계적으로 생각하여 답변을 생성하도록 유도하는 프롬프트를 사용합니다."
)
summarize_before_rerank_toggle = st.sidebar.checkbox(
    "리랭킹 전 문서 요약", value=False,
    help="리랭커가 처리하기 전에 각 문서를 OpenAI를 이용해 요약합니다. (OpenAI API 비용 발생)"
)
use_fact_checker_toggle = st.sidebar.checkbox(
    "Fact Checker 사용 (Upstage API)", value = False,
    help="답변 생성 후 Upstage Groundedness Check를 수행합니다. 점수가 낮으면 질문 재작성을 시도할 수 있습니다. (Upstage API 비용 발생)"
)
use_gpt_scoring_toggle = st.sidebar.checkbox(
    f"GPT 자동 평가 ({DEFAULT_GPT_SCORING_MODEL})", value=False,
    help=f"생성된 답변에 대해 {DEFAULT_GPT_SCORING_MODEL}를 사용하여 관련성, 충실도 등을 평가합니다. (OpenAI API 비용 발생)"
)
evaluate_multiturn_no_context = st.sidebar.checkbox(
    f"no_context 시스템에서 multiturn 평가", value=False,
    help=f"multiturn 질문을 no_context 시스템에서 평가할 때 사용합니다(OpenAI API 비용 발생가능)"
)
st.sidebar.markdown("---")


# --- Main Area ---
# Show current setting
col_sum1, col_sum2 = st.columns(2)
with col_sum1:
    st.caption(f"**현재 설정:** Embedding: `{selected_embedding_alias}` | LLM: `{selected_llm}` | Reranker: `{selected_reranker_method_default if use_reranker_default and selected_reranker_method_default else '사용 안 함'}`")
with col_sum2:
    st.caption(f"**부가 기능:** 멀티턴: `{'ON' if use_query_multiturn_toggle else 'OFF'}` | CoT: `{'ON' if use_cot_toggle else 'OFF'}` | 요약: `{'ON' if summarize_before_rerank_toggle else 'OFF'}` | FactCheck: `{'ON' if use_fact_checker_toggle else 'OFF'}` | GPT평가: `{'ON' if use_gpt_scoring_toggle else 'OFF'}`")


# Load vector DB
try:
    vectorstore = load_vector_store(selected_embedding_alias)
except Exception as e:
    st.error(f"메인 앱: 벡터 DB 로드 중 심각한 오류 발생 - {e}")
    vectorstore = None

# Load document


# 세션 상태 초기화
#query_db : query를 저장하는 DB
#turn_num : turn 횟수 저장
#history : 이전 대화 한줄 요약
if "query_db" not in st.session_state: st.session_state.query_db = []
if "turn_num" not in st.session_state: st.session_state.turn_num = 0
if 'evaluation_details' not in st.session_state: st.session_state.evaluation_details = {}
if 'aggregated_scores' not in st.session_state: st.session_state.aggregated_scores = {}
if "history" not in st.session_state: st.session_state.history = ""
if "history_word" not in st.session_state: st.session_state.history_word = []


if vectorstore:
    st.markdown("### 질문 입력")
    question_input = st.text_input(
        "질문을 입력하세요:",
        placeholder="예: 배추의 주요 병충해는 무엇인가요?",
        key="main_question_input_area",
        label_visibility="collapsed"
    )
    
    button_cols = st.columns(3)
    with button_cols[0]:
        run_button_clicked = st.button("질문하기", key="single_run_button", use_container_width=True, type="primary")
    with button_cols[1]:
        if st.button("새 대화 시작", key="reset_multiturn_button", help="멀티턴 대화 기록을 초기화합니다.", use_container_width=True):
            #query_multiturn varaible initialize
            st.session_state.query_db = []
            st.session_state.turn_num = 0
            #history initialize
            st.session_state.history = ""
            st.session_state.history_word = []
            st.success("새 대화가 시작되었습니다. 이전 대화 기록이 초기화되었습니다.")
    with button_cols[2]:
        with st.popover("대화 기록 보기", use_container_width=True):
            st.markdown("##### 현재까지의 대화 기록")
            if st.session_state.history or st.session_state.history_word:
                st.text_area("요약 내용:", value=st.session_state.history, height=150, disabled=True, label_visibility="collapsed")
                st.write("**추출된 핵심 단어:**")
                st.info(f"`{', '.join(st.session_state.history_word)}`")
            else:
                st.info("표시할 대화 기록이 없습니다. 'history 요약 멀티턴'을 활성화하고 대화를 시작해주세요.")


    if run_button_clicked and question_input:
        st.markdown("---")
        st.header("단일 실행 결과")
        
        current_question_for_pipeline = question_input
        
        if use_query_multiturn_toggle:
            st.session_state.query_db.append(question_input)
            st.session_state.turn_num += 1
            if st.session_state.turn_num > 1:
                st.write("멀티턴 질문 재구성 중...")
                with st.spinner("이전 질문을 바탕으로 질문을 재구성하고 있습니다..."):
                    try:
                        current_question_for_pipeline = query_reformulation(
                            st.session_state.query_db,
                            question_input,
                            st.session_state.turn_num
                        )
                        if current_question_for_pipeline != question_input:
                             st.info(f"재구성된 질문 (멀티턴): {current_question_for_pipeline}")
                        else:
                             st.info("멀티턴: 현재 질문을 그대로 사용합니다.")
                    except Exception as e:
                        st.error(f"멀티턴 질문 재구성 실패: {e}")
                        current_question_for_pipeline = question_input
            else:
                st.write("첫 번째 질문입니다. 멀티턴 재구성을 건너뜁니다.")
                current_question_for_pipeline = question_input
        else:
            st.session_state.query_db = [question_input]
            st.session_state.turn_num = 1
            current_question_for_pipeline = question_input


        #use_history_multirun_toggle일 경우  history + query로 재생성
        if use_history_multiturn_toggle and st.session_state.history_word:
            keywords_str = ", ".join(st.session_state.history_word)
            # 현재 파이프라인에 전달될 질문(current_question_for_pipeline)에 키워드를 추가
            current_question_for_pipeline = f"{current_question_for_pipeline} (이전 대화 핵심 단어들은 {keywords_str})"
            st.info(f"재구성된 검색용 질문 : {current_question_for_pipeline}")

        cost_flags = []
        if selected_embedding_alias == 'openai': cost_flags.append("OpenAI Embedding")
        if summarize_before_rerank_toggle and use_reranker_default : cost_flags.append("OpenAI Summarization")  
        if selected_llm.startswith('gpt'): cost_flags.append(f"OpenAI LLM ({selected_llm})")
        if use_fact_checker_toggle: cost_flags.append("Upstage FactCheck")
        if use_gpt_scoring_toggle: cost_flags.append(f"OpenAI Scoring ({DEFAULT_GPT_SCORING_MODEL})")
        if use_query_multiturn_toggle and st.session_state.turn_num > 1 and current_question_for_pipeline != question_input:
            cost_flags.append("OpenAI query Multiturn Rewrite")
        if use_history_multiturn_toggle : cost_flags.append("OpenAI history Multiturn Rewrite")
        if cost_flags:
            st.info(f"API 비용 발생 가능 항목: {', '.join(cost_flags)}")

        final_response_text = "오류: 처리 중 문제 발생"
        final_docs_used_for_response = []



        try:
            with st.spinner("RAG 파이프라인 실행 중... 잠시만 기다려주세요."):
                final_response_text, final_docs_used_for_response = run_rag_pipeline(
                    question=current_question_for_pipeline,
                    vectorstore=vectorstore,
                    retriever_k=retriever_k_value,
                    use_reranker=use_reranker_default,
                    reranker_method=selected_reranker_method_default,
                    reranker_top_k=reranker_top_k_value,
                    summarize_before_rerank=summarize_before_rerank_toggle,
                    llm_model_name=selected_llm,
                    use_cot=use_cot_toggle,
                    run_id="SingleRun"
                )

            average_fact_check_score = None
            if use_fact_checker_toggle and final_docs_used_for_response and final_response_text and "오류:" not in final_response_text:
                st.write("Fact Checking 수행 중... (Upstage API)")
                with st.spinner("답변 내용의 사실성을 검증하고 있습니다..."):
                    sentences_for_fc = sentence_split(final_response_text)
                    if sentences_for_fc:
                        try:
                            average_fact_check_score, _ = fact_checker(sentences_for_fc, final_docs_used_for_response)
                            st.info(f"Fact Check 평균 점수: {average_fact_check_score:.4f}")
                            if 0.3 < average_fact_check_score < 0.7: # 임계값 예시
                                st.warning("Fact Check 점수가 낮아 질문을 재작성하여 다시 시도합니다. (OpenAI API 사용)")
                                rewritten_q_for_fc_retry = rewrite_question_single(current_question_for_pipeline)
                                if rewritten_q_for_fc_retry != current_question_for_pipeline:
                                    st.write(f"재작성된 질문 (FactCheck Retry): {rewritten_q_for_fc_retry}")
                                    with st.spinner("재작성된 질문으로 RAG 파이프라인 재실행 중..."):
                                        final_response_text, final_docs_used_for_response = run_rag_pipeline(
                                            question=rewritten_q_for_fc_retry, vectorstore=vectorstore,
                                            retriever_k=retriever_k_value, use_reranker=use_reranker_default,
                                            reranker_method=selected_reranker_method_default, reranker_top_k=reranker_top_k_value,
                                            summarize_before_rerank=summarize_before_rerank_toggle,
                                            llm_model_name=selected_llm, use_cot=use_cot_toggle,
                                            run_id="RewriteFactCheck"
                                        )
                                else:
                                    st.warning("   질문 재작성에 변화가 없어 원본 질문 결과를 사용합니다.")
                        except Exception as e_fc:
                            st.error(f"Fact Checking 중 오류 발생: {e_fc}")
                    else:
                        st.write("   Fact Check를 위한 문장이 없거나, 점수가 임계치 범위 밖이어서 재시도를 건너뜁니다.")

            st.markdown("### AI 응답")
            st.markdown(final_response_text)

            if use_gpt_scoring_toggle and final_docs_used_for_response and final_response_text and "오류:" not in final_response_text:
                st.write(f" GPT 자동 평가 중... ({DEFAULT_GPT_SCORING_MODEL})")
                mock_ground_truth_for_single_run = st.text_input(
                    "GPT 평가용 Ground Truth (선택 사항, 입력 후 Enter):",
                    key="gt_input_single_run",
                    value="Ground truth를 제공하지 않았습니다."
                )
                
                context_str_for_scoring = format_docs(final_docs_used_for_response)
                with st.spinner(f"{DEFAULT_GPT_SCORING_MODEL} 모델을 사용하여 답변을 평가하고 있습니다..."):
                    try:
                        combined_scores_result = get_combined_score(
                            query=current_question_for_pipeline,
                            response=final_response_text,
                            context=context_str_for_scoring,
                            ground_truth=mock_ground_truth_for_single_run
                        )
                        st.markdown("#### GPT 자동 평가 점수")
                        st.json(combined_scores_result)
                    except Exception as e_gpt_score:
                        st.error(f"GPT 자동 평가 중 오류 발생: {e_gpt_score}")

            with st.expander("참고한 문서 (최종 답변 생성에 사용됨)", expanded=False):
                if final_docs_used_for_response:
                    for i, doc_item in enumerate(final_docs_used_for_response):
                        st.markdown(f"**문서 {i+1}**: *출처: {doc_item.metadata.get('file_name', 'N/A')} (인덱스: {doc_item.metadata.get('index', 'N/A')})*")
                        st.text_area(
                            label=f"문서 {i+1} 내용 보기",
                            value=doc_item.page_content,
                            height=150,
                            key=f"final_doc_content_{i}"
                        )
                else:
                    st.write("최종 답변 생성에 사용된 문서가 없습니다.")

            #use_history_multiturn이 체크 되어 있을 경우 가장 마지막 단계에 history 생성 후 저장
            if use_history_multiturn_toggle and final_response_text and "오류:" not in final_response_text:
                st.write(f" history 생성 중... ")
                st.session_state.history, st.session_state.history_word = summarize_conversation(st.session_state.history, current_question_for_pipeline, final_response_text)


        except Exception as e_single_run:
            st.error(f"단일 실행 처리 중 심각한 오류 발생: {e_single_run}")
            import traceback
            st.error(traceback.format_exc())


    # --- Evaluation Set Handling (평가셋 사용한 모델 조합 평가) ---
    st.sidebar.markdown("---")
    st.sidebar.header("3. 모델 성능 평가 (평가셋 사용)")
    st.sidebar.caption("현재 선택된 조합으로 업로드된 평가셋 전체에 대한 성능을 측정합니다.")

    uploaded_eval_file = st.sidebar.file_uploader(
        "평가셋 파일 업로드 (.json)", type=['json'], key="eval_set_file_uploader"
    )
    
    if st.sidebar.button("현재 설정으로 평가 실행", key="set_eval_run_button") and uploaded_eval_file:
        st.markdown("---")
        st.header("모델 성능 평가 결과 (평가셋)")

        
        #uploaded_eval_file의 json 파일을 파싱하여 저장. multiturn인 경우와 아닌 경우 구분하여 파싱
        if use_query_multiturn_toggle or use_history_multiturn_toggle :
            evaluation_set_data = load_evaluation_set(uploaded_eval_file, True)
        else : 
            evaluation_set_data = load_evaluation_set(uploaded_eval_file, False)

        #evaluation_set_data가 존재할 경우
        if evaluation_set_data:
            st.session_state.evaluation_details = {}
            st.session_state.aggregated_scores = {}

            # 사용자가 선택한 단일 조합으로 평가 리스트를 구성합니다.
            eval_embeddings_for_set = [selected_embedding_alias]
            eval_use_reranker_for_set = use_reranker_default
            if eval_use_reranker_for_set:
                reranker_list_to_iterate_set = [selected_reranker_method_default]
            else:
                reranker_list_to_iterate_set = [None]
            
            total_combinations = 1 # 항상 단일 조합만 평가
            total_questions_in_set = len(evaluation_set_data)
            st.info(f"선택된 조합(Embedding: `{selected_embedding_alias}`, Reranker: `{selected_reranker_method_default if use_reranker_default else '사용 안 함'}`)으로 평가를 시작합니다.")
            st.info(f"총 {total_questions_in_set}개 질문에 대해 평가를 실행합니다.")
            
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            runs_completed_count = 0
            total_runs_for_set = total_combinations * total_questions_in_set

            #선택한 임베딩 모델 종류에 대해
            for emb_alias_set in eval_embeddings_for_set:
                st.write(f"평가셋용 임베딩 모델 '{emb_alias_set}' 로딩 중...")
                current_vectorstore_set = load_vector_store(emb_alias_set)
                if not current_vectorstore_set:
                    st.error(f"평가셋 평가 중단: '{emb_alias_set}' 벡터 DB 로드 실패.")
                    runs_completed_count += len(reranker_list_to_iterate_set) * total_questions_in_set
                    progress_bar.progress(min(1.0, runs_completed_count / total_runs_for_set) if total_runs_for_set > 0 else 0.0)
                    continue

                for reranker_method_set in reranker_list_to_iterate_set:
                    reranker_display_name_set = '사용 안 함' if not eval_use_reranker_for_set or reranker_method_set is None else reranker_method_set
                    combination_key_set = (emb_alias_set, reranker_display_name_set)
                    
                    # session state initialize
                    st.session_state.aggregated_scores[combination_key_set] = {'fact_check_scores': [], 'gpt_scores_list': [], 'f1_scores': []}
                    st.session_state.evaluation_details[combination_key_set] = []
                    
                    #query initialize
                    st.session_state.query_db = []
                    st.session_state.turn_num = 0
                   
                    #history initialize
                    st.session_state.history = ""
                    st.session_state.history_word = []
                    st.session_state.question_num = 0


                    st.subheader(f"조합 평가 중: Embedding: {emb_alias_set}, Reranker: {reranker_display_name_set}")
                    
                    #model set에 대한 quetion 반복
                    #여기에 M item이 존재할 경우, 숫자가 바뀌기 전까지 계속해서 multiturn으로 진행하고 바뀌면 multiturn 끝내도록 변경.
                    prev_M_index = None
                    for q_idx, eval_item in enumerate(evaluation_set_data):
                        #현재 평가셋의 속성을 들고 온다
                        eval_question_text = eval_item['question']
                        question_input = eval_question_text
                        eval_ground_truth_text = eval_item['answer']

                        if use_query_multiturn_toggle or use_history_multiturn_toggle or evaluate_multiturn_no_context : 
                            # M 이전게 다른거거나 q_idx가 0이면 리셋하도록 ###
                            if st.session_state.question_num%2 == 0 :
                                #query_multiturn varaible initialize
                                st.session_state.query_db = []
                                st.session_state.turn_num = 0
                                #history initialize
                                st.session_state.history = ""
                                st.session_state.history_word = []

                            st.session_state.question_num += 1

                        run_id_set_item = f"Set_Emb:{emb_alias_set}_Rerank:{reranker_display_name_set}_Q:{q_idx+1}"
                        
                        progress_text.text(f"진행률: {runs_completed_count+1}/{total_runs_for_set} - {run_id_set_item}")
                        
                        #eval_response_item 등의 로컬 변수 초기화
                        current_question_for_pipeline = eval_question_text
                        eval_response_item = "오류"
                        eval_docs_used_item = []
                        avg_fc_score_item = None
                        f1_score_item = None
                        gpt_scores_item = None     
                            
                        
                        ########## multiturn query 수정 파트########

                        if use_query_multiturn_toggle:
                            st.session_state.query_db.append(eval_question_text)
                            st.session_state.turn_num += 1
                            if st.session_state.turn_num > 1:
                                st.write("멀티턴 질문 재구성 중...")
                                with st.spinner("이전 질문을 바탕으로 질문을 재구성하고 있습니다..."):
                                    try:
                                        current_question_for_pipeline = query_reformulation(
                                            st.session_state.query_db,
                                            eval_question_text,
                                            st.session_state.turn_num
                                        )
                                        if current_question_for_pipeline != eval_question_text:
                                            st.info(f"재구성된 질문 (멀티턴): {current_question_for_pipeline}")
                                        else:
                                            st.info("멀티턴: 현재 질문을 그대로 사용합니다.")
                                    except Exception as e:
                                        st.error(f"멀티턴 질문 재구성 실패: {e}")
                                        current_question_for_pipeline = eval_question_text
                            else:
                                st.write("첫 번째 질문입니다. 멀티턴 재구성을 건너뜁니다.")
                                current_question_for_pipeline = eval_question_text
                        else:
                            st.session_state.query_db = [eval_question_text]
                            st.session_state.turn_num = 1
                            current_question_for_pipeline = eval_question_text

                        # history multiturn 평가를 할 경우 history 키워드 + query로 재생성
                        if use_history_multiturn_toggle:
                            if st.session_state.question_num % 2 == 0 and st.session_state.history_word:
                                keywords_str = ", ".join(st.session_state.history_word)
                                
                                if not evaluate_multiturn_no_context:
                                    current_question_for_pipeline = f"{current_question_for_pipeline} (이전 대화 핵심 단어들은 {keywords_str})"
                                    st.write(f"재구성된 검색용 질문 (History 키워드 추가): {current_question_for_pipeline}")
                                
                                # GPT 평가용 질문 재구성은 검색용 질문과 별개로, 원본 질문과 history를 사용
                                eval_question_text = f"이전 대화의 내용인 \"{st.session_state.history}\"에 대해 \"{question_input}\"가 현재 질문이다. 응답에 대한 의도 평가 시 이를 고려하여라."
                                st.write(f"재구성된 평가용 질문(멀티턴): {eval_question_text}")

                        # multiturn이 아닐 때 또는 멀티턴 history가 있을 때 평가
                        if (use_history_multiturn_toggle is False and use_query_multiturn_toggle is False) or st.session_state.history != "" :

                            try:
                                # run_rag_pipeline 호출 시 멀티턴은 적용하지 않음 (평가셋 질문은 독립적)
                                # CoT는 사이드바 설정 따름
                                #eval_response_item이 AI 응답
                                #current_question_for_pipeline이 바꾼 query
                                
                                eval_response_item, eval_docs_used_item = run_rag_pipeline(
                                    question=current_question_for_pipeline, vectorstore=current_vectorstore_set,
                                    retriever_k=retriever_k_value, use_reranker=eval_use_reranker_for_set,
                                    reranker_method=reranker_method_set, reranker_top_k=reranker_top_k_value,
                                    summarize_before_rerank=summarize_before_rerank_toggle,
                                    llm_model_name=selected_llm, use_cot=use_cot_toggle,
                                    run_id=run_id_set_item
                                )
                                
                                if use_fact_checker_toggle and eval_docs_used_item and eval_response_item and "오류:" not in eval_response_item:
                                    sentences_fc_set = sentence_split(eval_response_item)
                                    if sentences_fc_set:
                                        avg_fc_score_item, _ = fact_checker(sentences_fc_set, eval_docs_used_item)
                                if avg_fc_score_item is not None:
                                    st.session_state.aggregated_scores[combination_key_set]['fact_check_scores'].append(avg_fc_score_item)
                                
                                if eval_docs_used_item:
                                    f1_score_item = compute_evalset_f1(eval_item, eval_docs_used_item)
                                    if f1_score_item is not None:
                                        st.session_state.aggregated_scores[combination_key_set]['f1_scores'].append(f1_score_item)


                                # 평가셋 이용시 무조건 평가하도록 수정
                                if eval_docs_used_item and eval_response_item and "오류:" not in eval_response_item:
                                    context_str_set = format_docs(eval_docs_used_item)
                                    gpt_scores_item = get_combined_score(eval_question_text, eval_response_item, context_str_set, eval_ground_truth_text)
                                
                                if gpt_scores_item and "Error" not in gpt_scores_item: # 오류가 없는 경우에만 추가
                                    st.session_state.aggregated_scores[combination_key_set]['gpt_scores_list'].append(gpt_scores_item)
                                
                                elif gpt_scores_item and "Error" in gpt_scores_item:
                                    st.warning(f"GPT Scoring Error for {run_id_set_item}: {gpt_scores_item['Error']}")


                                st.session_state.evaluation_details[combination_key_set].append({
                                    "question": question_input, "ground_truth": eval_ground_truth_text,
                                    "response": eval_response_item, "fact_check_score": avg_fc_score_item,
                                    "f1_score": f1_score_item, "gpt_scores": gpt_scores_item,
                                    "used_docs": eval_docs_used_item
                                })

                                
                            except Exception as e_set_item:
                                st.error(f"[{run_id_set_item}] 평가 실행 중 오류: {e_set_item}")
                                st.session_state.evaluation_details[combination_key_set].append({
                                    "question": question_input, "ground_truth": eval_ground_truth_text,
                                    "response": f"오류 발생: {e_set_item}", "fact_check_score": None,
                                    "f1_score": None, "gpt_scores": {"Error": str(e_set_item)},
                                    "used_docs": []
                                })
                            finally:
                                if use_history_multiturn_toggle and use_query_multiturn_toggle : 
                                    runs_completed_count += 2
                                else : 
                                    runs_completed_count += 1
                                progress_bar.progress(min(1.0, runs_completed_count / total_runs_for_set) if total_runs_for_set > 0 else 0.0)


                        #use_history_multiturn이 체크 되어 있을 경우 가장 마지막 단계에 history 생성 후 저장
                        #item[name]이 unrelate이면 single turn을 의미하므로 아닌 경우만 history 생성
                        if use_history_multiturn_toggle and eval_response_item and "오류:" not in eval_response_item and eval_item["name"] != ["unrelate"]:
                            st.write(f" history 생성 중... ")
                            #평가셋의 ground truth로 생성.
                            st.session_state.history, st.session_state.history_word = summarize_conversation(st.session_state.history, question_input, eval_ground_truth_text)
                    
                    progress_text.text(f"진행률: {runs_completed_count}/{total_runs_for_set} - 조합 완료: Emb: {emb_alias_set}, Rerank: {reranker_display_name_set}")

            progress_bar.progress(1.0)
            progress_text.text(f"평가 완료! 총 {runs_completed_count}/{total_runs_for_set} 실행 완료.")

            # --- 평가셋 기반 평균 점수 표시 ---
            st.markdown("---")
            st.subheader("평가셋 기반 모델 조합별 평균 점수")
            
            avg_results_display_list = []
            for (emb_key, rer_key), scores_data_dict in st.session_state.aggregated_scores.items():
                num_total_questions = total_questions_in_set
                
                # Fact Check 평균
                avg_fc = np.mean(scores_data_dict['fact_check_scores']) if scores_data_dict['fact_check_scores'] else None
                # F1 Score 평균
                avg_f1 = np.mean(scores_data_dict['f1_scores']) if scores_data_dict['f1_scores'] else None
                
                # GPT Scores 평균 (각 항목별로)
                avg_gpt_scores_summary = {}
                if scores_data_dict['gpt_scores_list']:
                    # 첫 번째 유효한 GPT 점수 항목에서 키들을 가져옴
                    first_valid_gpt_score = next((s for s in scores_data_dict['gpt_scores_list'] if s and "Error" not in s), None)
                    if first_valid_gpt_score:
                        for score_metric_key in first_valid_gpt_score.keys():
                            if "Error" in score_metric_key: continue # 에러 키는 제외
                            valid_metric_scores = [
                                s[score_metric_key] for s in scores_data_dict['gpt_scores_list'] 
                                if s and "Error" not in s and score_metric_key in s and isinstance(s[score_metric_key], (int, float))
                            ]
                            if valid_metric_scores:
                                avg_gpt_scores_summary[f"Avg. {score_metric_key}"] = np.mean(valid_metric_scores)
                            else:
                                avg_gpt_scores_summary[f"Avg. {score_metric_key}"] = None
                
                num_successful_runs = len([s for s in st.session_state.evaluation_details.get((emb_key, rer_key), []) if "오류 발생:" not in s["response"]])

                result_line = f"**Embedding: {emb_key}, Reranker: {rer_key}** (성공: {num_successful_runs}/{num_total_questions})"
                avg_results_display_list.append(result_line)
                if avg_fc is not None: avg_results_display_list.append(f"  - Avg. Fact Check: {avg_fc:.4f}")
                if avg_f1 is not None: avg_results_display_list.append(f"  - Avg. F1 Score (Doc Retrieval): {avg_f1:.4f}")
                if avg_gpt_scores_summary:
                    for gpt_key, gpt_avg_val in avg_gpt_scores_summary.items():
                        avg_results_display_list.append(f"  - {gpt_key}: {gpt_avg_val:.4f}" if gpt_avg_val is not None else f"  - {gpt_key}: N/A")
                avg_results_display_list.append("---")

            st.markdown("\n".join(avg_results_display_list))

            # --- 평가셋 기반 상세 결과 표시 ---
            st.markdown("---")
            st.subheader("평가셋 기반 모델 조합별 상세 결과")
            if not st.session_state.evaluation_details:
                st.write("상세 결과가 없습니다.")
            else:
                for (emb_k_detail, rer_k_detail), details_list_for_combo in st.session_state.evaluation_details.items():
                    expander_label = f"Embedding: {emb_k_detail}, Reranker: {rer_k_detail} (총 {len(details_list_for_combo)}개 질문)"
                    with st.expander(expander_label):
                        if not details_list_for_combo:
                            st.write("이 조합에 대한 결과가 없습니다.")
                            continue
                        for item_idx, detail_item in enumerate(details_list_for_combo):
                            st.markdown(f"**질문 {item_idx+1}:** {detail_item['question']}")
                            st.markdown(f"**정답 (Ground Truth):** {detail_item['ground_truth']}")
                            st.markdown(f"**AI 응답:**")
                            st.write(detail_item['response'])
                            if detail_item.get('fact_check_score') is not None:
                                st.caption(f"Fact Check Score: {detail_item['fact_check_score']:.4f}")
                            if detail_item.get('f1_score') is not None:
                                st.caption(f"F1 Score (Doc Retrieval): {detail_item['f1_score']:.4f}")
                            if detail_item.get('gpt_scores') is not None:
                                if "Error" in detail_item['gpt_scores']:
                                    st.error(f"GPT Score Error: {detail_item['gpt_scores']['Error']}")
                                else:
                                    st.caption("GPT 자동 평가 점수:")
                                    st.json(detail_item['gpt_scores'])
                            
                            used_docs_detail = detail_item.get('used_docs', [])
                            if used_docs_detail:
                                with st.expander(f"참고 문서 보기 ({len(used_docs_detail)}개)", expanded=False):
                                    for doc_j, doc_obj_detail in enumerate(used_docs_detail):
                                        st.markdown(f"**문서 {doc_j+1}**: *출처: {doc_obj_detail.metadata.get('file_name', 'N/A')} (인덱스: {doc_obj_detail.metadata.get('index', 'N/A')})*")
                                        st.text_area(
                                            label=f"문서 {doc_j+1} 내용", value=doc_obj_detail.page_content, height=100,
                                            key=f"detail_doc_content_{emb_k_detail}_{rer_k_detail}_{item_idx}_{doc_j}"
                                        )
                            else:
                                st.caption("참고한 문서 없음")
                            st.markdown("---") # 각 질문 항목 구분


    # 이전 평가 결과 표시 로직 (세션 상태에 결과가 남아있는 경우)
    elif st.session_state.evaluation_details and not uploaded_eval_file : # 버튼 안눌렸지만 이전 결과 있을때
        st.markdown("---")
        st.subheader("이전 평가 결과 (평가셋 기반)")
        st.info("새로운 평가를 실행하려면 평가셋 파일을 업로드하고 '현재 설정으로 평가 실행' 버튼을 클릭하세요.")

else: # vectorstore 로드 실패 시
    st.error(f"벡터 DB를 로드할 수 없습니다. 사이드바에서 선택된 임베딩 모델에 대한 FAISS 인덱스가 '{FAISS_DB_PATH / selected_embedding_alias}' 경로에 올바르게 생성되어 있는지 확인해주세요.")

st.markdown("---")
st.caption("RAG System (Refactored Version) | Powered by Streamlit, LangChain, OpenAI, Upstage, FAISS, HuggingFace")