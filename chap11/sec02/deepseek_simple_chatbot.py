from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 로컬 Ollama에 실행 중인 DeepSeek 모델을 연결합니다.
llm = ChatOllama(model="deepseek-r1:8b")

# 대화의 첫 시스템 메시지(역할/규칙)를 미리 저장합니다.
messages = [
    SystemMessage("너는 사용자의 질문에 한국어로 답변해야 한다.")
]

while True:
    # 사용자 입력을 받아 앞뒤 공백을 제거합니다.
    user_input = input("You\t: ").strip()

    # 종료 명령어를 입력하면 루프를 종료합니다.
    if user_input in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    # 사용자 메시지를 히스토리에 추가한 뒤 모델 스트리밍 응답을 시작합니다.
    messages.append(HumanMessage(user_input))
    response = llm.stream(messages)

    # 스트리밍 chunk를 합쳐 최종 AI 메시지 1개로 복원합니다.
    ai_message = None
    for chunk in response:
        print(chunk.content, end="")
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk
    print('')

    # deepseek-r1 출력의 내부 추론 태그(</think>) 뒤 실제 답변만 저장합니다.
    # 태그가 없을 때 예외가 날 수 있어 split 결과 길이를 확인합니다.
    parts = ai_message.content.split("</think>", 1)
    message_only = parts[1].strip() if len(parts) > 1 else ai_message.content.strip()

    # 다음 턴 문맥 유지를 위해 AI 답변을 히스토리에 추가합니다.
    messages.append(AIMessage(message_only))
