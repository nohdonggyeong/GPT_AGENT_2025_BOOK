"""
- 시간대별로 화자를 구분하는 함수 speaker_diarization

          start      end  speaker_id  duration
number                                        
0         0.993   30.204  SPEAKER_00    29.211
1        32.414   41.391  SPEAKER_01     8.977
2        41.611   42.455  SPEAKER_00     0.844
3        41.645   42.708  SPEAKER_01     1.063
4        42.674   44.024  SPEAKER_00     1.350
5        45.813   67.109  SPEAKER_01    21.296
6        67.227   82.786  SPEAKER_00    15.559
7        84.659  102.564  SPEAKER_01    17.905
8       103.492  117.532  SPEAKER_00    14.040
9       119.759  138.676  SPEAKER_01    18.917
10      139.351  168.967  SPEAKER_00    29.616
11      170.907  192.321  SPEAKER_01    21.414
12      192.322  193.689  SPEAKER_00     1.367
13      192.760  193.503  SPEAKER_01     0.743
14      193.823  216.571  SPEAKER_00    22.748
15      218.579  237.783  SPEAKER_01    19.204
16      238.103  238.677  SPEAKER_00     0.574
17      238.188  239.352  SPEAKER_01     1.164
18      239.858  240.651  SPEAKER_00     0.793
19      240.297  240.989  SPEAKER_01     0.692
20      240.989  251.334  SPEAKER_00    10.345
21      253.696  271.550  SPEAKER_01    17.854
22      272.140  304.557  SPEAKER_00    32.417
23      306.970  326.022  SPEAKER_01    19.052
24      326.360  335.472  SPEAKER_00     9.112
25      337.548  355.925  SPEAKER_01    18.377
26      356.245  363.974  SPEAKER_00     7.729
27      365.982  381.423  SPEAKER_01    15.441
28      382.165  383.734  SPEAKER_00     1.569
29      385.726  397.184  SPEAKER_01    11.458
30      397.623  399.108  SPEAKER_00     1.485
31      401.437  406.601  SPEAKER_01     5.164
32      407.326  408.338  SPEAKER_00     1.012
33      410.093  417.451  SPEAKER_01     7.358
34      417.755  421.231  SPEAKER_00     3.476
35      423.644  429.348  SPEAKER_01     5.704
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

if __name__ == "__main__":
    audio_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/audio/싼기타_비싼기타.mp3"
    stt_output_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타.csv"
    rttm_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타.rttm"
    rttm_csv_file_path = "/Users/donggyeong/develop/now/GPT_AGENT_2025_BOOK/chap05/sec03/싼기타_비싼기타_rttm.csv"

    df_rttm = speaker_diarization(
        audio_file_path,
        rttm_file_path,
        rttm_csv_file_path
    )

    print(df_rttm)
