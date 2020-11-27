import os
import tensorflow as tf
import numpy as np
import math
from convolution import conv2d
from matplotlib import pyplot as plt
import librosa


###
# Postprocessing
###
def spectrogram_to_audio(spectrogram):
	"""

	:param spectrogram:
	:param :
	:return:
	"""

	# i-Stft parameters
	n_fft = 4096
	hop_length = 256
	sample_rate = 22050
	w_length = 1024

	# Tensorflow's inverse stft
	tf.signal.inverse_stft(spectrogram, frame_length=w_length, frame_step=hop_length, fft_length=n_fft, window_fn=tf.signal.hann_window, name=None)

	# Librosa's inverse sftf
	librosa.istft(spectrogram, hop_length=hop_length, win_length=win_length, window='hann', center=True, dtype=None, length=None)
	# Librosa's inverse stft from melspectrogram
	librosa.feature.inverse.mel_to_audio(spectrogram, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None, dtype=np.float32)

	pass