import librosa.core as lc
import numpy as np
import scipy
import math
import cmath

def reduction(input_path,output_path):

    # パラメータ
    amp_up_threshold=0.05
    fixed_amp=0.05
    amp_down_threshold=0.001


    # データのインプット
    data, sampleRate = lc.load(input_path, sr=44100, duration=10, mono=True)

    # 短区間フーリエ変換
    f,t,stftMat = scipy.signal.stft(data,fs=sampleRate,nperseg=4096)

    # 振幅値を計算
    amplist=np.abs(stftMat)

    # 閾値以上の振幅を変換
    conversion_list=list(zip(*np.where(amplist>amp_up_threshold)))
    for i in conversion_list:
        # 極座標変換
        abs,rad =cmath.polar(stftMat[i])
        abs=fixed_amp
        stftMat[i]=cmath.rect(abs,rad)

    # 閾値以下の振幅を0に変換
    conversion_list=list(zip(*np.where(amplist<amp_down_threshold)))
    for i in conversion_list:
        # 極座標変換
        abs,rad =cmath.polar(stftMat[i])
        abs=0
        stftMat[i]=cmath.rect(abs,rad)

    #フィルタ処理
    # 7000Hz以上をカット
    freqlist=np.where(f>7500)
    stftMat[np.amin(freqlist):,:]=0
    # 人が不快に感じる周波数をカット
    freqlist=np.where((4000>f)&(f>2000))
    stftMat[np.amin(freqlist):np.amax(freqlist),:]=0

    # 逆フーリエ変換
    dummy,iStftMat = scipy.signal.istft(stftMat,sampleRate)

    # 出力
    scipy.io.wavfile.write(output_path, 44100, iStftMat)

if __name__ == '__main__':
    reduction("/Users/kaburakimidorihitoshi/Documents/huk/noise/model/voice.wav","/Users/kaburakimidorihitoshi/Documents/huk/noise/model/output.wav")

