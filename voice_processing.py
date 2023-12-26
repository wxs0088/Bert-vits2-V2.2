import os
from pathlib import Path
import librosa
from scipy.io import wavfile
import numpy as np
import whisper
import argparse
from langconv import *


def split_long_audio(model, filepath, save_dir="raw", out_sr=44100) -> str:
    '''将长音源wav文件分割为短音源文件，返回短音源文件存储路径path'''
    # 短音频文件存储路径
    save_dir = os.path.join(os.path.dirname(filepath), save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 分割文件
    print(f'分割文件{filepath}...')
    # 获取文件名
    filename = os.path.basename(filepath).split('.wav')[0]
    result = model.transcribe(filepath, word_timestamps=True, task="transcribe", beam_size=5, best_of=5)
    segments = result['segments']
    wav, sr = librosa.load(filepath, sr=None, offset=0, duration=None, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav = 0.98 * wav / peak
    wav2 = librosa.resample(wav, orig_sr=sr, target_sr=out_sr)
    wav2 /= max(wav2.max(), -wav2.min())
    for i, seg in enumerate(segments):
        start_time = seg['start']
        end_time = seg['end']
        wav_seg = wav2[int(start_time * out_sr):int(end_time * out_sr)]
        wav_seg_name = "{}_{}.wav".format(filename, str(i))  # 修改名字
        i += 1
        out_fpath = os.path.join(save_dir, wav_seg_name)
        wavfile.write(out_fpath, rate=out_sr, data=(wav_seg * np.iinfo(np.int16).max).astype(np.int16))
    return save_dir


def transcribe_one(audio_path):  # 使用whisper语音识别
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)
    # 繁体转简体
    txt = result.text
    txt = Converter('zh-hans').convert(txt)

    fileName = os.path.basename(audio_path)
    print(f'{fileName}:{lang}——>{txt}')
    return txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFilePath', type=str, help="干声源音频wav文件的全路径")
    parser.add_argument('listFileSavePath', type=str, help=".list文件存储全路径")
    parser.add_argument('--shortFilesPath', type=str,
                        help="已经分割好了的短音频的存储目录全路径，用于当分割好之后再次运行时配置")
    opt = parser.parse_args()
    print(f'参数：{opt}')

    model = whisper.load_model("medium")
    # 将长音源分割成短音源文件
    if not opt.shortFilesPath:
        save_dir = split_long_audio(model, opt.inputFilePath)
    else:
        save_dir = opt.shortFilesPath

    # 为每个短音频文件提取文字内容，生成.lab文件和filelists目录下的.list文件
    if not os.path.exists(opt.listFileSavePath):
        file = open(opt.listFileSavePath, "w")
        file.close()
    print('提取文字内容...')
    files = os.listdir(save_dir)
    spk = os.path.basename(os.path.dirname(opt.inputFilePath))
    for file in files:
        if not file.endswith('.wav'):
            continue
        text = transcribe_one(os.path.join(save_dir, file))
        with open(opt.listFileSavePath, 'a', encoding="utf-8") as wf:
            wf.write(f"{os.path.join(save_dir, file).replace('raw','wavs')}|lilith|ZH|{text}\n")

    print('音频预处理完成！')

