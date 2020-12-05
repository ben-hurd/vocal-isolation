import numpy as np
import librosa
import wavio


###
# Postprocessing
###
def spectrogram_to_audio(melspectrogram, filename):
	"""

	:param spectrogram:
	:param :
	:return:
	"""

	n_fft = 4096
	hop_length = 256
	sample_rate = 22050
	w_length = 1024

	audio = librosa.feature.inverse.mel_to_audio(melspectrogram, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=w_length, window='hann', center=True, pad_mode='reflect', power=2.0, n_iter=64, length=None, dtype=np.float32)

	wavio.write(filename, audio, sample_rate, sampwidth=3)

def main():

	return


if __name__ == '__main__':
    main()