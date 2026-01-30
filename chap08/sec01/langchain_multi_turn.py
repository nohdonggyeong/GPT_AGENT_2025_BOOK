# from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(api_key=api_key)
llm = ChatOpenAI(model="gpt-4o")

# def get_ai_response(messages):
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     temperature=0.9,
    #     messages=messages,
    # )
    # return response.choices[0].message.content

messages = [
    # {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
    SystemMessage("너는 사용자를 도와주는 상담사야.")
]

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    messages.append(
        # {"role": "user", "content": user_input}
        HumanMessage(user_input)
    )

    # ai_response = get_ai_response(messages=messages)
    ai_response = llm.invoke(messages)

    messages.append(
        # {"role": "assistant", "content": ai_response}
        ai_response
    )
    print("AI: " + ai_response.content)

'''
사용자:  안녕? 내 이름은 노동경이야.
AI: 안녕하세요, 노동경님! 반갑습니다. 오늘 어떻게 도와드릴까요?
사용자: 내가 누구게?
AI: 노동경님이라고 소개해주셨는데, 이름으로만은 정확히 어떤 분인지 알기 어렵네요. 혹시 어떤 힌트를 주실 수 있나요? 또는 어떤 이야기를 나누고 싶으신가요?
사용자: 미국에서 인기있는 연예인은 누구야?
AI: 미국에서 인기 있는 연예인은 시기에 따라 다를 수 있지만, 몇몇 인물들은 꾸준히 높은 인기를 유지하고 있습니다. 예를 들어, 가수 테일러 스위프트(Taylor Swift)와 비욘세(Beyoncé)는 음악적으로 큰 영향력을 가지고 있고, 배우로는 드웨인 존슨(Dwayne "The Rock" Johnson)과 제니퍼 로렌스(Jennifer Lawrence)가 대중에게 사랑받고 있습니다. 또한, 최근에는 배우 겸 프로듀서인 리한나(Rihanna)도 음악 외의 활동으로 주목받고 있죠. 이 외에도 다양한 분야에서 활약하는 스타들이 많습니다. 관심 있는 분야가 있으면 그에 맞춰 더 구체적인 정보를 드릴 수도 있습니다.
사용자: 한국에서는 어때?
AI: 한국에서는 K-pop과 드라마 등의 영향으로 많은 연예인들이 인기를 끌고 있습니다. K-pop 그룹 중에서는 방탄소년단(BTS), 블랙핑크(Blackpink), 세븐틴(Seventeen) 등이 국제적으로도 많은 사랑을 받고 있습니다. 

드라마와 영화에서는 배우 송중기, 이정재, 전지현, 박보검, 김혜수 등이 대중에게 큰 인기를 얻고 있습니다. 

또한 최근 예능 프로그램에서 활약 중인 유재석, 강호동 같은 진행자들도 꾸준히 사랑받고 있습니다. 한국 연예계는 매우 다양한 분야에서 많은 인재들이 활동하고 있어, 관심 있는 분야에 따라 더 많은 정보를 드릴 수 있습니다.
사용자:  exit
'''