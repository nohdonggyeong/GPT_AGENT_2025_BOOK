# 실습: 이미지 분석 + TTS용 영어 듣기 평가 문제 생성

# 이미지 파일을 Base64로 인코딩하는 유틸
from glob import glob
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

# .env에서 API 키 로드 후 클라이언트 생성
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 단일 이미지로 퀴즈 1개 생성(재시도 포함)
def image_quiz(image_path, n_trial=0, max_trial=3):
    if n_trial >= max_trial:
        raise Exception("Failed to generate a quiz.")
    
    base64_image = encode_image(image_path)
    quiz_prompt = """
    제공한 이미지를 바탕으로, 다음과 같은 양식으로 퀴즈를 만들어 주세요.
    정답은 (1)~(4) 중 하나만 해당하도록 출제하세요.
    토익 리스닝 문제 스타일로 문제를 만들어 주세요.
    아래는 예시입니다.
    ---- 예시 ----

    Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
    - (1) 베이커리에서 사람들이 빵을 사는 모습이 담겨 있습니다.
    - (2) 맨 앞에 서 있는 사람은 빨간색 셔츠를 입었습니다.
    - (3) 기차를 타기 위해 줄을 서 있는 사람들이 있습니다.
    - (4) 점원은 노란색 티셔츠를 입었습니다.

    Listening: Which of the following descriptions of the image is incorrect?
    - (1) It shows people buying bread at a bakery.
    - (2) The person standing at the front is wearing a red shirt.
    - (3) There are people lining up to take a train.
    - (4) The clerk is wearing a yellow T-shirt.

    정답: (4) 점원은 노란색 티셔츠가 아닌 파란색 티셔츠를 입었습니다.
    (주의: 정답은 (1)~(4) 중 하나만 선택하도록 출제하세요.)
    =====
    """

    # 텍스트 지시문 + 이미지 데이터를 함께 전달
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": quiz_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
    except Exception as e:
        # 실패 시 재시도
        print("failed\n" + e)
        return image_quiz(image_path, n_trial+1)
    
    content = response.choices[0].message.content

    # 형식 검증(영문 Listening 문장이 있는지 확인)
    if "Listening:" in content:
        return content, True
    else:
        return image_quiz(image_path, n_trial+1)

# 단일 이미지로 1문제 생성 테스트
q = image_quiz("/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap06/sec02/busan_dive.jpg")
print(q)

# 여러 이미지로 문제집/스크립트 생성

txt = ''
eng_dict = []
no = 1
for g in glob('/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap06/sec02/*.jpg'):
    q, is_suceed = image_quiz(g)

    if not is_suceed:
        continue

    divider = f'## 문제 {no}\n\n'
    print(divider)

    txt += divider
    filename = os.path.basename(g)
    txt += f'![image]({filename})\n\n'

    print(q)
    txt += q + '\n\n----------\n\n'

    # 마크다운 문제집 저장
    with open('/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap06/sec02/image_quiz.md', 'w', encoding='utf-8') as f:
        f.write(txt)
    
    # 영어 스크립트만 추출해 TTS용 JSON 구성
    eng = q.split('Listening: ')[1].split('정답:')[0].strip()

    eng_dict.append({
        'no': no,
        'eng': eng,
        'img': filename
    })

    # JSON 파일로 저장
    with open('/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap06/sec02/image_quiz_eng.json', 'w', encoding='utf-8') as f:
        json.dump(eng_dict, f, ensure_ascii=False, indent=4)

    no += 1
