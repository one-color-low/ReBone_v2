import os
from os.path import join

from .models.cyclegan_vc import CycleGAN
from .speech_tools import *

src_speaker = 'my_voice'
trg_speaker = 'natori'
model_name = 'cyclegan_vc2'
pretrain_dir = 'pretrain_data'

def convert_voice(wav, sampling_rate=22050, num_mcep=36, frame_period=5.0, n_frames=128):
    print('Loading cached data...')
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A \
    = load_pickle(join(pretrain_dir, 'cache', '{}{}.p'.format(src_speaker, num_mcep)))
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B \
    = load_pickle(join(pretrain_dir, 'cache', '{}{}.p'.format(trg_speaker, num_mcep)))

    model = CycleGAN(num_features=num_mcep, batch_size=1, mode='test')
    model.load(join(pretrain_dir, '{}_100000.ckpt'.format(model_name)))

    print('Generating Validation Data B from A...')
    wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
    f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                    mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
    coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
    coded_sp_transposed = coded_sp.T
    coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
    coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='A2B')[0]
    if coded_sp_converted_norm.shape[1] > len(f0):
        coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
    coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean
    coded_sp_converted = coded_sp_converted.T
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
    wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                             frame_period=frame_period)

    return wav_transformed

if __name__ == '__main__':
    wav, _ = librosa.load("./audio.wav")
    converted_voice = convert_voice(wav)
    librosa.output.write_wav('./test.wav',converted_voice,sr=22050)
