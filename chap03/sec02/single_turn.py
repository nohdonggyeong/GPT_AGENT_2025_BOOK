from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=[
            {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
            {"role": "user", "content": user_input},
        ],
    )

    print("AI: " + response.choices[0].message.content)

'''
사용자: 안녕? 내 이름은 노동경이야.
AI: 안녕하세요, 노동경님! 만나서 반갑습니다. 어떻게 도와드릴까요?
사용자: 내 이름이 뭘까?
AI: 죄송하지만, 저는 당신의 이름을 알 수 없습니다. 대신 어떻게 불리고 싶은지 알려주시면 그렇게 부르겠습니다. 무엇을 도와드릴까요?
사용자: exit
'''