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

	# Tensorflow's inverse stft
	tf.signal.inverse_stft(spectrogram, frame_length=, frame_step=, fft_length=None, window_fn=tf.signal.hann_window, name=None)

	# Librosa's inverse sftf
	librosa.istft(spectrogram, hop_length=None, win_length=None, window='hann', center=True, dtype=None, length=None)
	# Librosa's inverse stft from melspectrogram
	librosa.feature.inverse.mel_to_audio(spectrogram, sr=22050, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0, n_iter=32, length=None, dtype=np.float32)

	pass