import os
import tensorflow as tf
import numpy as np
import math
import librosa
import librosa.display
import matplotlib.pyplot as plt
import musdb
import skimage.io


###
# Preprocessing
###

def make_spectrograms():
	"""
	Generates all of the spectrograms from the musdb18 dataset.
	:return: None
	"""
	
	mus_train = musdb.DB(root="data/musdb18", subsets="train")
	
	# Creating the spectogram images from the training data
	for track in mus_train:

		# Converting samples to target rate of 22050
		original_sr = track.rate
		target_sr = 22050
		track_data = librosa.resample(librosa.to_mono(track.audio.T), orig_sr=original_sr, target_sr=target_sr, res_type='kaiser_best', fix=True, scale=False)
		vocal_data = librosa.resample(librosa.to_mono(track.targets['vocals'].audio.T), orig_sr=original_sr, target_sr=target_sr, res_type='kaiser_best', fix=True, scale=False)

		# Length of frame; 66150 should be 3 seconds (appears as 6 seconds on graph)
		len_frame = target_sr*3
		num_frames = int(len(track_data)/len_frame)

		# Saving each frame as a spectrogram (and putting track in mix folders and vocals in vocals folder)
		for frame in range(num_frames):
			make_spectrogram(track_data[frame * len_frame : frame * len_frame + len_frame], "mix/" + track.name + "-" + str(frame) + ".png", True)
			make_spectrogram(vocal_data[frame * len_frame : frame * len_frame + len_frame], "vocals/" + track.name + "-" + str(frame) + ".png", True)

	return


def make_spectrogram(inputs, filename, is_train):
	"""
	Makes an individual spectrogram and saves it in the corresponding folder
	:param inputs: array of sample values of a given frame
	:param filename: string, to be the name of the saved spectrogram file
	:param train: boolean specifying if training is occuring
	"""

	# Stft parameters
	n_fft = 4096
	hop_length = 256
	n_mels = 128
	f_min = 20
	f_max = 11637
	# f_max = 14000
	sample_rate = 22050
	w_length = 1024

	# Librosa melspectrogram
	mels = librosa.feature.melspectrogram(inputs, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=w_length)

	# Getting rid of edges of figure
	figure = plt.figure(figsize=(500, 600), dpi=1)
	axis = plt.subplot(1, 1, 1)
	plt.axis('off')
	plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

	S_db = librosa.power_to_db(mels, ref=np.max)
	librosa.display.specshow(S_db, fmin=f_min, fmax=f_max)

	extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
	
	# Saving figure
	if is_train:
		plt.savefig("data/spectrograms/train/" + filename, bbox_inches=extent, pad_inches=0)
	else:
		plt.savefig("data/spectrograms/test/" + filename, bbox_inches=extent, pad_inches=0)

	# Close plots for memory purposes
	plt.clf()
	plt.close()


def main():

	# Generate spectrogram files from dataset (only will do the training files at the moment)
	make_spectrograms()

	return


if __name__ == '__main__':
    main()