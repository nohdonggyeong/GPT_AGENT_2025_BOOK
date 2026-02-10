# chap11~12 발표 노트 (코드별 쉬운 설명 + 개조식)

## 1) chap11/sec02/deepseek_simple_chatbot.py
- 목적: DeepSeek-R1 기반 CLI 대화 챗봇의 최소 구조 이해
- 핵심 흐름:
  1. `ChatOllama`로 로컬 모델 연결
  2. `SystemMessage`로 역할 고정
  3. 사용자 입력을 `HumanMessage`로 누적
  4. `llm.stream(messages)`로 토큰 단위 출력
  5. 최종 AI 응답을 `AIMessage`로 다시 저장
- 설명 포인트:
  - 스트리밍 응답은 `chunk`를 합쳐 최종 1개 메시지로 관리
  - 다음 턴 문맥 유지의 핵심은 `messages` 리스트 누적
  - `</think>` 태그 처리 로직은 모델별 출력 포맷 대응 예시

## 2) chap11/sec03/retriever.py
- 목적: RAG용 검색기(retriever) + 답변 체인 + 질의 보강 체인 구성
- 핵심 흐름:
  1. 임베딩 모델 준비 (`text-embedding-3-large`)
  2. Chroma 벡터스토어 로드
  3. `as_retriever(k=3)`로 상위 3개 문서 검색
  4. `document_chain`으로 문서 기반 답변 생성
  5. `query_augmentation_chain`으로 질문 명확화
- 설명 포인트:
  - 검색 품질은 "질문 품질"에 크게 좌우됨
  - 그래서 원문 질의 + 보강 질의를 같이 활용
  - 프롬프트에서 `MessagesPlaceholder`를 써 대화 맥락 유지

## 3) chap11/sec03/rag_deepseek.py
- 목적: Streamlit UI에서 RAG 챗봇을 실사용 형태로 연결
- 핵심 흐름:
  1. 사용자 질문 입력
  2. 질의 보강 체인 실행
  3. 관련 문서 검색
  4. 검색 문서를 컨텍스트로 답변 스트리밍
  5. 답변을 세션 상태에 저장
- 설명 포인트:
  - `st.session_state["messages"]`가 대화 메모리 역할
  - 문서 원문을 `expander`로 보여줘 근거 확인 가능
  - 답변 생성/검색/메모리 흐름을 UI에서 한 번에 시연 가능

## 4) chap12/sec01/langgraph_simple_chatbot.ipynb
- 목적: LangGraph의 가장 단순한 상태 그래프 챗봇 구조 이해
- 셀별 요약:
  1. `langgraph` 설치
  2. `ChatOpenAI` 모델 로드
  3. `State(TypedDict)` 정의 (`messages` + `add_messages`)
  4. `generate` 노드 구현
  5. `START -> generate -> END` 엣지 연결
  6. 그래프 시각화
  7. `invoke`로 단일 호출
  8. 응답 state 재투입해 대화 연속성 확인
  9. `stream`으로 토큰 스트리밍 확인
- 설명 포인트:
  - LangGraph 핵심은 "State + Node + Edge"
  - `add_messages`는 메시지 덮어쓰기 대신 누적 동작

## 5) chap12/sec02/langgraph_memory.py
- 목적: LangGraph에 메모리(checkpointer)를 붙여 스레드 단위 대화 유지
- 핵심 흐름:
  1. `MemorySaver()` 생성
  2. `graph_builder.compile(checkpointer=memory)`
  3. `configurable.thread_id`로 대화 세션 식별
  4. 루프에서 사용자 입력마다 `graph.stream(..., stream_mode="values")`
- 설명 포인트:
  - thread_id가 같으면 이전 대화 맥락 자동 복원
  - 상태 저장이 분리되어 멀티 세션 대응 확장 가능

## 6) chap12/sec03/langgraph_tools.ipynb
- 목적: LangGraph 에이전트에 도구(tool) 호출 루프 추가
- 셀별 요약:
  1. 모델 준비 (`gpt-4o`)
  2. `State`/그래프 빌더 준비
  3. `@tool` 함수 2개 정의
     - `get_current_time`: 타임존 시각 조회
     - `get_web_search`: DuckDuckGo 검색
  4. `model.bind_tools(tools)`로 모델에 도구 스키마 연결
  5. `generate` 노드 추가
  6. `BasicToolNode` 구현 (AI의 tool_calls 실행)
  7. `route_tools`로 분기 (`tool_calls` 있으면 tools 노드)
  8. `generate <-> tools` 루프 구성 후 컴파일
  9. 시간 질의/리서치형 프롬프트 스트리밍 실행
- 설명 포인트:
  - 에이전트의 본질: "생각(generate) -> 행동(tools) -> 재생각(generate)"
  - `ToolMessage`를 다시 state에 넣어 후속 추론 가능
  - 조건부 엣지가 있어야 도구 호출 여부에 따라 분기 가능

---

# 발표용 개조식 (슬라이드 바로 옮기기)

## A. 전체 학습 목표
- LangChain 기반 기본 챗봇 구조 이해
- RAG 파이프라인(질문 보강 + 검색 + 근거 기반 답변) 구현
- LangGraph로 상태 기반 에이전트 구조 확장
- 메모리/도구 호출을 붙여 실전형 챗봇으로 발전

## B. chap11 핵심 메시지
- 단순 대화형 챗봇에서 시작해 RAG로 고도화
- 검색 품질 개선을 위해 Query Augmentation 적용
- Streamlit으로 사용자 경험(UI)까지 통합
- "문서 근거 제시 가능한 챗봇"을 구현했다는 점 강조

## C. chap12 핵심 메시지
- LangGraph는 상태(State) 중심 오케스트레이션 도구
- Memory(checkpointer)로 대화 연속성 보장
- Tool 노드 + 조건부 라우팅으로 에이전트 루프 완성
- 단일 LLM 호출에서 "판단-행동" 구조로 확장됨

## D. 기술적 인사이트
- 프롬프트 설계가 검색/답변 품질에 직접 영향
- 상태 관리 전략(메시지 누적, thread_id)이 안정성 좌우
- 도구 호출은 필수 기능만 작게 시작하는 것이 유지보수에 유리

## E. 한계와 개선 방향
- 한계:
  - 모델 환각 가능성
  - 검색 결과 품질 편차
  - 도구 실행 실패 시 예외 처리 미흡
- 개선:
  1. 근거 문서 출처 자동 인용 포맷 고정
  2. 재시도/타임아웃/폴백 모델 추가
  3. 평가셋 기반 RAG 정량 평가 도입

## F. 데모 시나리오(권장)
1. 챗봇 기본 대화 (chap11/sec02)
2. RAG 질문 + 근거 문서 확인 (chap11/sec03)
3. LangGraph 메모리 유지 확인 (chap12/sec02)
4. 현재 시간/웹검색 도구 호출 시연 (chap12/sec03)

## G. 마무리 한 줄
- "단순 대화형 LLM에서 시작해, 상태/메모리/도구를 갖춘 에이전트형 시스템으로 단계적으로 확장했다."
