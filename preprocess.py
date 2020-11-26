import os
import tensorflow as tf
import numpy as np
import math
import librosa
import librosa.display
import matplotlib.pyplot as plt
###
# Preprocessing
###
def get_data(train_filename, test_filename):
	"""

	:param :
	:param :
	:return:
	"""
	### Read in data and convert audio to array
	train_data, train_sample_rate = librosa.load(train_filename, sr=44100, mono=False, dtype=np.float32, res_type='kaiser_best')
	test_data, test_sample_rate = librosa.load(test_filename, sr=44100, mono=False, dtype=np.float32, res_type='kaiser_best')

	# ?Convert to mono? - Stereo is ideal and would be much cooler
	# train_data = librosa.to_mono(train_data)
	# test_data = librosa.to_mono(test_data)

	# ?Trim silence? - I think it could mess up the alignment of different stems and is unnecessary
	# y = librosa.effects.trim(y, top_db=60, ref=<function amax>, frame_length=2048, hop_length=512)

	# ?Downsample from 44.1kHz to 22.05 kHz?
	sample_rate = 44100
	train_data = librosa.resample(train_data, orig_sr=sample_rate, target_sr=22050, res_type='kaiser_best', fix=True, scale=False)
	test_data = librosa.resample(train_data, orig_sr=sample_rate, target_sr=22050, res_type='kaiser_best', fix=True, scale=False)
	make_spectrogram(train_data[0])
	make_spectrogram(test_data[0])


	### Extract the inputs and labels (full mixes & actual vocal stems)
	# train_mixes = train_data[0]
	# train_vocals = train_data[4]
	# test_mixes = test_data[0]
	# test_vocals = test_data[4]

	return train_data, train_data, test_data, test_data

def make_spectrogram(inputs):
	"""
	:param :
	:param :
	:return:
	"""
	n_fft = 1024
	hop_length = 256
	n_mels = 40
	f_min = 20
	f_max = 8000
	sample_rate = 16000

	# Tensorflow sftf
	X = tf.signal.stft(signals=inputs, frame_length=1024, frame_step=256, fft_length=None, window_fn=tf.signal.hann_window, pad_end=False, name=None)

	# Librosa stft
	X = librosa.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True, dtype=None, pad_mode='reflect')
	# stft_magnitude, stft_phase = librosa.magphase(stft)
	# stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)
	# mel_spec = librosa.feature.melspectrogram(clip, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, sr=sample_rate, power=1.0, fmin=20, fmax=16000)
	# mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

	# Librosa mel-stft
	fig, ax = plt.subplots()
	S = librosa.feature.melspectrogram(inputs, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
	S_db = librosa.power_to_db(S, ref=np.max)
	img = librosa.display.specshow(S_db, y_axis='mel', x_axis='time', ax=ax)
	ax.set(title='Mel spectrogram display')
	fig.colorbar(img, ax=ax, format="%+2.f dB")
	plt.show()

    

def main():
    """

    :return: None
    """
    train_mixes, train_vocals, test_mixes, test_vocals = get_data('data/skateboard-p-flip.wav','data/skateboard-p-flip-instrumental.wav')


    return


if __name__ == '__main__':
    main()