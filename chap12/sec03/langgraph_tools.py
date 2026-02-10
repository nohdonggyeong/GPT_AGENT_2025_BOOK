"""LangGraph + Tool calling demo script.

이 파일은 `langgraph_tools.ipynb`의 코드를 발표/실행용 단일 파이썬 스크립트로 정리한 버전입니다.

실행 전 준비:
- OPENAI_API_KEY 환경 변수가 설정되어 있어야 합니다.
- 필요 패키지: langchain-openai, langgraph, langchain-community, duckduckgo-search, pytz

사용 예시:
- 기본 데모(서울 현재 시각):
    python chap12/sec03/langgraph_tools.py
- 기사 작성 데모(툴 검색 반복):
    python chap12/sec03/langgraph_tools.py --mode article
- 두 데모 모두 실행:
    python chap12/sec03/langgraph_tools.py --mode all
"""

from __future__ import annotations

import argparse
import json
import pytz

from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()

# -----------------------------------------------------------------------------
# 1) 상태(State) 정의
# -----------------------------------------------------------------------------
class State(TypedDict):
    """LangGraph에서 공유할 상태 객체.

    messages 키는 대화 기록을 담고,
    add_messages를 통해 새 메시지가 기존 리스트에 누적됩니다.
    """

    messages: Annotated[list, add_messages]


# -----------------------------------------------------------------------------
# 2) 툴(Tool) 정의
# -----------------------------------------------------------------------------
@tool
def get_current_time(timezone: str, location: str) -> str:
    """지정한 타임존의 현재 시각을 문자열로 반환합니다."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timezone} ({location}) 현재 시각 {now}"
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"


@tool
def get_web_search(query: str, search_period: str = "m") -> str:
    """DuckDuckGo 웹 검색 결과를 반환합니다.

    search_period:
    - d: 최근 하루
    - w: 최근 일주일
    - m: 최근 한 달
    - y: 최근 1년
    """
    wrapper = DuckDuckGoSearchAPIWrapper(time=search_period)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, results_separator=";\n")

    # 발표 시 콘솔에서 검색 흐름이 보이도록 로그를 남깁니다.
    print("\n-------- WEB SEARCH --------")
    print(f"query={query}")
    print(f"period={search_period}")

    searched = search.invoke(query)
    for i, result in enumerate(searched.split(";\n"), start=1):
        print(f"{i}. {result}")

    return searched


# -----------------------------------------------------------------------------
# 3) ToolNode 구현: 모델이 요청한 툴을 실제로 실행하는 노드
# -----------------------------------------------------------------------------
class BasicToolNode:
    """마지막 AIMessage의 tool_calls를 읽어 툴 실행 후 ToolMessage를 생성합니다."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool_item.name: tool_item for tool_item in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No message found in input")

        message = messages[-1]
        outputs = []

        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    # ToolMessage content는 문자열이어야 하므로 JSON 문자열로 감쌉니다.
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )

        # 기존 대화 + 툴 실행 결과를 합쳐 다음 노드로 전달합니다.
        return {"messages": messages + outputs}


# -----------------------------------------------------------------------------
# 4) 그래프 라우팅 함수
# -----------------------------------------------------------------------------
def route_tools(state: State):
    """마지막 메시지에 tool_calls가 있으면 tools 노드로, 없으면 종료(END)합니다."""
    if isinstance(state, list):
        ai_message = state[-1]
    else:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError(f"tool_edge 입력 상태에서 메시지를 찾을 수 없습니다: {state}")
        ai_message = messages[-1]

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# -----------------------------------------------------------------------------
# 5) 그래프 생성 함수
# -----------------------------------------------------------------------------
def build_graph(model_name: str = "gpt-4o", temperature: float = 0.01):
    """모델/툴/노드를 연결한 LangGraph 인스턴스를 생성합니다."""
    model = ChatOpenAI(model=model_name, temperature=temperature)
    tools = [get_current_time, get_web_search]
    model_with_tool = model.bind_tools(tools)

    def generate(state: State):
        # 현재 메시지 히스토리를 바탕으로 모델 응답 1개를 생성합니다.
        return {"messages": [model_with_tool.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("tools", BasicToolNode(tools=tools))

    # 시작 -> generate
    graph_builder.add_edge(START, "generate")

    # generate 결과에 tool_calls가 있으면 tools, 아니면 종료
    graph_builder.add_conditional_edges(
        "generate",
        route_tools,
        {"tools": "tools", END: END},
    )

    # tools 실행 후 다시 generate로 돌아가 후속 응답 생성
    graph_builder.add_edge("tools", "generate")

    return graph_builder.compile()


# -----------------------------------------------------------------------------
# 6) 데모 실행 함수
# -----------------------------------------------------------------------------
def run_time_demo(graph) -> None:
    """간단 질의 데모: '지금 서울 몇 시야?'"""
    print("\n===== DEMO 1: 현재 시각 질의 =====")
    inputs = [HumanMessage(content="지금 서울 몇 시야?")]

    for msg, _metadata in graph.stream({"messages": inputs}, stream_mode="messages"):
        if isinstance(msg, AIMessageChunk):
            print(msg.content, end="")
    print("\n")


def run_article_demo(graph, about: str = "서울 월드컵 경기장 잔디 문제") -> None:
    """심층 기사 작성 데모: 검색 도구를 반복 호출하며 기사 초안을 생성합니다."""
    print("\n===== DEMO 2: 심층 기사 작성 =====")

    prompt = f"""
너는 신문기자이다.
최근 {about}에 대해 비판하는 심층 분석 기사를 쓰려고 한다.

-최근 어떤 이슈가 있는지 검색하고, 사람들이 제일 관심 있어 할만 한 주제를 선정하고, 왜 선정했는지 말해줘.
-그 내용으로 원고를 작성하기 위한 목차를 만들고, 목차 내용을 채우기 위해 추가로 검색할 내용을 리스트로 정리해봐.
-검색할 리스트를 토대로 재검색해.
-목차에 있는 내용을 작성하기 위해 더 검색이 필요한 정보가 있는지 확인하고, 있다면 추가로 검색해.
-검색된 결과는 원하는 정보를 찾지 못했다면 다른 검색어로 재검색해도 좋아.

더 이상 검색할 내용이 없다면, 조선일보 신문 기사 형식으로 최종 기사를 작성한다.
제목, 부제, 리드문, 본문의 구성으로 작성한다. 본문 내용은 심층 분석 기사에 맞게 구체적이고 깊이 있게 작성해야 한다.
""".strip()

    inputs = [SystemMessage(content=prompt)]
    for msg, _metadata in graph.stream({"messages": inputs}, stream_mode="messages"):
        if isinstance(msg, AIMessageChunk):
            print(msg.content, end="")
    print("\n")


# -----------------------------------------------------------------------------
# 7) CLI 엔트리포인트
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph tool-calling demo")
    parser.add_argument(
        "--mode",
        choices=["time", "article", "all"],
        default="time",
        help="실행 모드 선택",
    )
    parser.add_argument("--model", default="gpt-4o", help="사용할 OpenAI 모델명")
    parser.add_argument("--temperature", type=float, default=0.01, help="모델 temperature")
    args = parser.parse_args()

    graph = build_graph(model_name=args.model, temperature=args.temperature)

    if args.mode in ("time", "all"):
        run_time_demo(graph)
    if args.mode in ("article", "all"):
        run_article_demo(graph)


if __name__ == "__main__":
    main()
