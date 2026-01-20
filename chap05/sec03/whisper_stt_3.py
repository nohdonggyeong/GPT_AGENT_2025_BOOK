"""
- STT 결과 파일과 화자 분리 결과 파일(RTTM) 결합하기

          start      end  ... duration                                               text
number                    ...                                                            
0         0.993   30.204  ...   29.211  지금부터 저랑 그 역할극을 합시다 역할극을 스탠딩 코미디 스타일로 할 건데 토론을 ...
1        32.414   41.391  ...    8.977  좋습니다. 그럼 제가 싼 기타로 시작하는 게 좋다는 입장을 맡아볼게요. 그럼 성형님...
2        41.611   42.455  ...    0.844                                                   
3        41.645   42.708  ...    1.063                                  네 맞아요. 준비 되셨나요?\n
4        42.674   44.024  ...    1.350                                    네 됐어요. 시작하시죠.\n
5        45.813   67.109  ...   21.296  좋아요.\n먼저 3기타로 시작하는게\n좋은 이유를 말씀드리겠습니다.\n초보자일때는\...
6        67.227   82.786  ...   15.559  아, 저는 지금 말에 어폐가 있다고 생각해요.\n왜냐하면 어차피 지금 비싼 기타로\...
7        84.659  102.564  ...   17.905  그런데 비싼 기타로 시작하면\n혹시라도 흠집이 나거나 실수할 때 부담이 더 크지 않...
8       103.492  117.532  ...   14.040  싼 기타를 뭐하러 또 삽니까?\n그리고 기타 실력이랑 흠집이랑은 아무 상관이 없어요...
9       119.759  138.676  ...   18.917  하하, 기타를 망치로 치진 않지만\n그래도 초보자들은 실수도 많고\n조심스럽게 다루...
10      139.351  168.967  ...   29.616  충분하죠. 아 맞아요. 그것도 맞는 말이에요. 쌍기타도 요새 품질이\n많이 좋아서\...
11      170.907  192.321  ...   21.414  하하\n역시 좋은 기타를 사면 책임감도\n더 커진다는 말씀이시군요 그래도 비싼 기타...
12      192.322  193.689  ...    1.367                                                   
13      192.760  193.503  ...    0.743                                                   
14      193.823  216.571  ...   22.748  아니에요. 왜냐하면 비싼 기타로 시작을 하면 연습할 때 기분이 안 나잖아요. 그러니...
15      218.579  237.783  ...   19.204  측면에서도 비싼 기타가 훨씬 도움이 된다고 생각합니다. 하하, 비싼 기타가 일종의 ...
16      238.103  238.677  ...    0.574                                                   
17      238.188  239.352  ...    1.164                                   결국은 본인이 선택한...\n
18      239.858  240.651  ...    0.793                                                   
19      240.297  240.989  ...    0.692                                                   
20      240.989  251.334  ...   10.345  근데 그게 비싼 기타를 치면 더 편해요.\n편하게 치는 게 중요하다고 하셨잖아요.\...
21      253.696  271.550  ...   17.854  맞아요. 비싼 기타는 확실히 소리도 좋고 연주하기 편하죠.\n하지만 초보자 입장에서...
22      272.140  304.557  ...   32.417  아 그 말에 대해 동의를 합니다.\n기타를 잘 못 치는 사람은\n쌍기타와 비싼 기타...
23      306.970  326.022  ...   19.052  하하, 쌍기타의 맛을 모르는 게 오히려 더 좋다는 말씀이시군요\n하지만 비싼 기타로...
24      326.360  335.472  ...    9.112  맞아요. 그래서 비싼 기타를 사야 되는 거예요.\n비싼 기타를 사면 꾸준히 연습을 ...
25      337.548  355.925  ...   18.377  그렇죠. 비싼 기타는 그만큼 소리도 좋고 연주할 때 기분도 좋으니까요.\n하지만 역...
26      356.245  363.974  ...    7.729   아, 맞아요.\n지금 얘기하신 거 보니까\n쌍기타가 더 좋다는 얘기는 안 하시네요.\n
27      365.982  381.423  ...   15.441  아마 제가 이긴 거 같아요. 그렇죠? 하하, 토론에서 이기고 지는 것보다는\n재밌게...
28      382.165  383.734  ...    1.569                                 어? 지금 도망가시는 건가요?\n
29      385.726  397.184  ...   11.458  아니요. 전혀 그렇지 않아요.\n전엔 그냥 즐거운 대화를 나누고 싶었을 뿐이에요.\...
30      397.623  399.108  ...    1.485                                                   
31      401.437  406.601  ...    5.164        아니요. 전혀 화나지 않았어요. 편하게 이야기 나누는 중이니까 걱정하세요.\n
32      407.326  408.338  ...    1.012                                        화난 것 같은데?\n
33      410.093  417.451  ...    7.358  네, 정말 괜찮습니다. 즐겁게 대화 나누고 있었어요.\n계속해서 이하고 나누고 싶으...
34      417.755  421.231  ...    3.476                  화나셨는데 굳이 더 할 필요 없죠.\n그만 잊지 마시죠.\n
35      423.644  429.348  ...    5.704  알겠습니다. 언제든 다시 이야기 나누고 싶으실 때 편하게 말씀해 주세요. 감사합니다.\n

[36 rows x 5 columns]
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