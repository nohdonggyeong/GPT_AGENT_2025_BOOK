"""
실습. 판다스로 문장 분석하고 화자 매칭하기
- 로컬에 설치한 Whisper 모델을 활용해 받아쓰기 함수 만들기

      start     end                                               text
0      1.00   11.00  지금부터 저랑 그 역할극을 합시다 역할극을 스탠딩 코미디 스타일로 할 건데 토론을 ...
1     12.00   17.00         그래서 좀 재밌고 자연스럽고 유머러스하게 저랑 대화를 하시면 돼요 자연스럽게
2     17.00   21.46                      그리고 주제는 쌍기타로 전기기타를 시작하는 게 좋으냐
3     21.46   24.36                        아니면 비싼 기타로 전기기타를 시작하는 게 좋으냐
4     24.36   26.00                                      이거를 입장을 나눠가지고
..      ...     ...                                                ...
119  414.54  417.00                      계속해서 이하고 나누고 싶으시면 편하게 말씀해주세요.
120  417.00  418.00                                               아니요.
121  418.00  420.00                                화나셨는데 굳이 더 할 필요 없죠.
122  420.00  423.00                                         그만 잊지 마시죠.
123  423.00  428.96    알겠습니다. 언제든 다시 이야기 나누고 싶으실 때 편하게 말씀해 주세요. 감사합니다.

[124 rows x 3 columns]
"""

import os
import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers.utils.import_utils as import_utils

os.environ["PATH"] += os.pathsep + r"/opt/homebrew/bin/ffmpeg/bin"
# TorchCodec import can fail on some macOS setups; force-disable to avoid crash.
import_utils._torchcodec_available = False

def whisper_stt(
    audio_file_path: str,      
    output_file_path: str = "./output.csv"
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,  # 청크별로 타임스탬프를 반환
        chunk_length_s=10,  # 입력 오디오를 10초씩 나누기
        stride_length_s=2,  # 2초씩 겹치도록 청크 나누기
    )

    result = pipe(audio_file_path)
    df = whisper_to_dataframe(result, output_file_path)

    return result, df


def whisper_to_dataframe(result, output_file_path):
    start_end_text = []

    for chunk in result["chunks"]:
        start = chunk["timestamp"][0]
        end = chunk["timestamp"][1]
        text = chunk["text"].strip()
        start_end_text.append([start, end, text])
        df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
        df.to_csv(output_file_path, index=False, sep="|")
    
    return df


if __name__ == "__main__":
    result, df = whisper_stt(
        "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.mp3", 
        "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타.csv", 
    )

    print(df)
