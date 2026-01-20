"""
- STT 결과 파일과 화자 분리 결과 파일(RTTM) 결합하기

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
from dotenv import load_dotenv
import torch
import torchaudio
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers.utils.import_utils as import_utils
from pyannote.audio import Pipeline

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

def speaker_diarization(
        audio_file_path: str,
        output_rttm_file_path: str,
        output_csv_file_path: str
):
    load_dotenv()
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HUGGING_FACE_TOKEN
    )

    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("mps is available")
    else:
        print("mps is not available")

    waveform, sr = torchaudio.load(audio_file_path)
    out = pipeline({"waveform": waveform, "sample_rate": sr})

    ann = out.speaker_diarization
    with open("싼기타_비싼기타.rttm", "w", encoding="utf-8") as rttm:
        ann.write_rttm(rttm)

    # 판다스 데이터프레임으로 변환
    df_rttm = pd.read_csv(
        output_rttm_file_path,  # rttm 파일 경로
        sep=' ',  # 구분자는 띄어쓰기
        header=None,  # 헤더는 없음
        names=['type', 'file', 'chnl', 'start', 'duration', 'C1', 'C2', 'speaker_id', 'C3', 'C4']
    )

    df_rttm['end'] = df_rttm['start'] + df_rttm['duration']

    df_rttm["number"] = None  # number 열 만들고 None으로 초기화
    df_rttm.at[0, "number"] = 0

    for i in range(1, len(df_rttm)):
        if df_rttm.at[i, "speaker_id"] != df_rttm.at[i-1, "speaker_id"]:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"] + 1
        else:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"]

    df_rttm_grouped = df_rttm.groupby("number").agg(
        start=pd.NamedAgg(column='start', aggfunc='min'),
        end=pd.NamedAgg(column='end', aggfunc='max'),
        speaker_id=pd.NamedAgg(column='speaker_id', aggfunc='first')
    )

    df_rttm_grouped["duration"] = df_rttm_grouped["end"] - df_rttm_grouped["start"]
    
    df_rttm_grouped.to_csv(
        output_csv_file_path,
        index=False,
        encoding='utf-8'
    )

    return df_rttm_grouped

def stt_to_rttm(
        audio_file_path: str,
        stt_output_file_path: str,
        rttm_file_path: str,
        rttm_csv_file_path: str,
        final_output_csv_file_path: str
    ):

    result, df_stt = whisper_stt(
        audio_file_path, 
        stt_output_file_path
    )

    df_rttm = speaker_diarization(
        audio_file_path,
        rttm_file_path,
        rttm_csv_file_path
    )

    df_rttm["text"] = ""

    for i_stt, row_stt in df_stt.iterrows():
        overlap_dict = {}
        for i_rttm, row_rttm in df_rttm.iterrows():
            overlap = max(0, min(row_stt["end"], row_rttm["end"]) - max(row_stt["start"], row_rttm["start"]))
            overlap_dict[i_rttm] = overlap
        
        max_overlap = max(overlap_dict.values())
        max_overlap_idx = max(overlap_dict, key=overlap_dict.get)

        if max_overlap > 0:
            df_rttm.at[max_overlap_idx, "text"] += row_stt["text"] + "\n"

    df_rttm.to_csv(
        final_output_csv_file_path,
        index=False,    # 인덱스는 저장하지 않음
        sep='|',
        encoding='utf-8'
    )
    return df_rttm


if __name__ == "__main__":
    audio_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.mp3"
    stt_output_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타.csv"
    rttm_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타.rttm"
    rttm_csv_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타_rttm.csv"
    final_csv_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타_final.csv"

    df_rttm = stt_to_rttm(
        audio_file_path,
        stt_output_file_path,
        rttm_file_path,
        rttm_csv_file_path,
        final_csv_file_path
    )

    print(df_rttm)