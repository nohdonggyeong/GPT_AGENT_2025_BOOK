from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from typing import List

from utils import save_state
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

filename = os.path.basename(__file__)
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)

llm = ChatOpenAI(model="gpt-4o")

class State(TypedDict):
    messages: List[AnyMessage | str]

def communicator(state: State):
    print("\n\n======== COMMUNICATOR ========")

    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI 팀의 커뮤니케이터로서,
        AI 팀의 진행 상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위해 대화를 나눈다.

        messages: {messages}
        """
    )

    system_chain = communicator_system_prompt | llm

    messages = state["messages"]

    inputs = {"messages": messages}

    gathered = None

    print('\nAI\t: ', end='')
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    
    messages.append(gathered)

    return {"messages": messages}

graph_builder = StateGraph(State)

graph_builder.add_node("communicator", communicator)

graph_builder.add_edge(START, "communicator")
graph_builder.add_edge("communicator", END)
graph = graph_builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))

state = State(
    messages = [
        SystemMessage(
            f"""
            너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가 팀이다.
            사용자가 사용하는 언어로 대화하라.

            현재 시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.

            """
        )
    ],
)

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n-------- MESSAGE COUNT\t', len(state["messages"]))
    save_state(current_path, state)