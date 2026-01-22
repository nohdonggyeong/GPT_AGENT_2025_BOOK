# Whisper로 전사, speaker-diarization으로 화자 분리하는 모듈

import os
from dotenv import load_dotenv
import torch
import torchaudio
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import transformers.utils.import_utils as import_utils
from pyannote.audio import Pipeline

os.environ["PATH"] += os.pathsep + r"/opt/homebrew/bin/ffmpeg/bin"
import_utils._torchcodec_available = False

def whisper_stt(
    audio_file_path: str,      
    output_file_path: str = "./output.csv"
):
    # 디바이스/정밀도 설정 (GPU 사용 시 fp16)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    # Whisper 모델 로드
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    # 토크나이저/특징 추출기 로드
    processor = AutoProcessor.from_pretrained(model_id)

    # 청크 단위 전사를 위한 파이프라인 구성
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        chunk_length_s=10,
        stride_length_s=2,
    )

    # 전사 실행 및 CSV로 변환
    result = pipe(audio_file_path)
    df = whisper_to_dataframe(result, output_file_path)

    return result, df


def whisper_to_dataframe(result, output_file_path):
    start_end_text = []

    # 청크별 타임스탬프와 텍스트를 테이블로 저장
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
    # Hugging Face 토큰 로드 및 파이프라인 생성
    load_dotenv()
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HUGGING_FACE_TOKEN
    )

    # 가용 디바이스 확인 및 적용
    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')

    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
        print("mps is available")
    else:
        print("mps is not available")

    # 다이어리제이션 수행 후 RTTM 저장
    waveform, sr = torchaudio.load(audio_file_path)
    out = pipeline({"waveform": waveform, "sample_rate": sr})

    ann = out.speaker_diarization
    with open("싼기타_비싼기타.rttm", "w", encoding="utf-8") as rttm:
        ann.write_rttm(rttm)

    # RTTM을 데이터프레임으로 변환
    df_rttm = pd.read_csv(
        output_rttm_file_path,
        sep=' ',
        header=None,
        names=['type', 'file', 'chnl', 'start', 'duration', 'C1', 'C2', 'speaker_id', 'C3', 'C4']
    )

    # 구간 종료 시각 계산
    df_rttm['end'] = df_rttm['start'] + df_rttm['duration']

    # 연속된 동일 화자를 하나의 구간 번호로 묶기
    df_rttm["number"] = None
    df_rttm.at[0, "number"] = 0

    for i in range(1, len(df_rttm)):
        if df_rttm.at[i, "speaker_id"] != df_rttm.at[i-1, "speaker_id"]:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"] + 1
        else:
            df_rttm.at[i, "number"] = df_rttm.at[i-1, "number"]

    # 구간 번호 기준으로 시작/끝/화자 집계
    df_rttm_grouped = df_rttm.groupby("number").agg(
        start=pd.NamedAgg(column='start', aggfunc='min'),
        end=pd.NamedAgg(column='end', aggfunc='max'),
        speaker_id=pd.NamedAgg(column='speaker_id', aggfunc='first')
    )

    # 구간 길이 계산 후 CSV 저장
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

    # 화자 구간별 텍스트 누적
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

    # 화자 구간+텍스트 결합 결과 저장
    df_rttm.to_csv(
        final_output_csv_file_path,
        index=False,
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
