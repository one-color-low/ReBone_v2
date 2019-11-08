import librosa.core as lc
import numpy as np
import scipy
from scipy import signal
import math
import cmath
import librosa

#調整パラメータ
# amp_max_smooth_filterのパラメータ
amp_up_threshold=0.1
fixed_amp=0.1

# amp_zero_filterのパラメータ
amp_down_threshold=0.003

# カットオフ周波数
fc=7500


def load_wav(input_path):
    data, sampleRate = librosa.load(input_path, sr=22050, duration=10, mono=True)
    return data,sampleRate

def save_wav(data,output_path):
    data=data.astype(np.float32)
    # librosa.output.write_wav(output_path,data,sr=22050)
    scipy.io.wavfile.write(output_path,22050,data)

# 短区間フーリエ変換
def STFT(data,sampleRate):
    freq,t,F_data = scipy.signal.stft(data,fs=sampleRate,nperseg=4096)
    return freq,t,F_data

# 短区間逆フーリエ変換
def STIFT(F_data,sampleRate):
    dummy,data = scipy.signal.istft(F_data,sampleRate)
    return data

# 閾値以上の振幅を固定値に変換するフィルター
def amp_max_smooth_filter(F_data):
    amplist=np.abs(F_data)
    # 閾値以上の振幅を変換
    conversion_list=list(zip(*np.where(amplist>amp_up_threshold)))
    for i in conversion_list:
        # 極座標変換
        abs,rad =cmath.polar(F_data[i])
        abs=fixed_amp
        F_data[i]=cmath.rect(abs,rad)
    return F_data

# 閾値以下の振幅を0に変換するフィルター
def amp_zero_filter(F_data):
    amplist=np.abs(F_data)
    conversion_list=list(zip(*np.where(amplist<amp_down_threshold)))
    for i in conversion_list:
        # 極座標変換
        abs,rad =cmath.polar(F_data[i])
        abs=0
        F_data[i]=cmath.rect(abs,rad)
    return F_data

# ローパスフィルター
def low_pass_filter(freq,F_data):
    # fc以上をカット
    freqlist=np.where(freq>fc)
    F_data[np.amin(freqlist):,:]=0
    return F_data

# バンドカットフィルター
def bandcut_filter(freq,F_data):
    # 人が不快に感じる周波数をカット
    freqlist=np.where((4000>freq)&(freq>2000))
    F_data[np.amin(freqlist):np.amax(freqlist),:]=0
    return F_data

# (基本使わない)時間領域でのバンドパスフィルター
def bandpass_filter(data,sampleRate):
    fp = np.array([2000, 8000])     #通過域端周波数[Hz]※ベクトル
    fs = np.array([50,15000 ])      #阻止域端周波数[Hz]※ベクトル
    gpass = 3                       #通過域端最大損失[dB]
    gstop = 40                      #阻止域端最小損失[dB]
    fn = sampleRate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band")           #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, data)                  #信号に対してフィルタをかける
    return y

# 周辺雑音除去フィルタ
def main_reduction(input_path,output_path):
    data,samplerate=load_wav(input_path)
    frequency,t,F_domain_data=STFT(data,samplerate)

    ####フィルター処理系#################################
    F_domain_data=amp_zero_filter(F_domain_data)
    ##################################################
    data=STIFT(F_domain_data,samplerate)
    save_wav(data,output_path)

# 高周波除去フィルタ
def main_lowpass(input_path, output_path):
    data,samplerate=load_wav(input_path)
    frequency,t,F_domain_data=STFT(data,samplerate)

    ####フィルター処理系#################################
    F_domain_data=low_pass_filter(frequency,F_domain_data)
    ##################################################
    data=STIFT(F_domain_data,samplerate)
    save_wav(data,output_path)


if __name__ == '__main__':
    main_reduction("/Users/kaburakimidorihitoshi/Documents/huk/noise/model/voice.wav","/Users/kaburakimidorihitoshi/Documents/huk/noise/model/output.wav")