import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import musdb
import sys
import zipfile
import gdown
import pickle
import os


###
# Preprocessing
###

def make_spectrograms(data, test):
	"""
	Generates all of the spectrograms from the musdb18 dataset.
	:return: None
	"""
	if data=="train":
		mus_data = musdb.DB(root="data/musdb18", subsets="train")
	else:
		mus_data = musdb.DB(root="data/musdb18", subsets="test")
	

	dictionary = {}
	dictionary["mix"] = []
	dictionary["vocals"] = []
	dictionary["instrumental"] = []

	# Creating the spectogram arrays from the training data
	num_tracks = len(mus_data)
	percent = 0.1

	for i, track in enumerate(mus_data):

		if i / num_tracks > percent:
			print(int(100 * percent), "%", "of " + data +  " data generated")
			percent += 0.1

		# Converting samples to target rate of 22050
		original_sr = track.rate
		target_sr = 22050
		mix_data = librosa.resample(librosa.to_mono(track.audio.T), orig_sr=original_sr, target_sr=target_sr, res_type='kaiser_best', fix=True, scale=False)
		vocal_data = librosa.resample(librosa.to_mono(track.targets['vocals'].audio.T), orig_sr=original_sr, target_sr=target_sr, res_type='kaiser_best', fix=True, scale=False)
		instrumental_data = librosa.resample(librosa.to_mono(track.targets['accompaniment'].audio.T), orig_sr=original_sr, target_sr=target_sr, res_type='kaiser_best', fix=True, scale=False)

		# Length of frame; 66150 should be 3 seconds (appears as 6 seconds on graph)
		len_frame = target_sr*3
		num_frames = int(len(mix_data)/len_frame)

		# Saving each frame as a spectrogram array (and putting track in mix folders and vocals in vocals folder)
		for frame in range(num_frames):
			dictionary["mix"].append(generate_spectrogram_array(mix_data[frame * len_frame : frame * len_frame + len_frame]))
			dictionary["vocals"].append(generate_spectrogram_array(vocal_data[frame * len_frame : frame * len_frame + len_frame]))
			dictionary["instrumental"].append(generate_spectrogram_array(instrumental_data[frame * len_frame : frame * len_frame + len_frame]))
			if test:
				pickle.dump(dictionary, open( "data/spectrograms/" + data + "-1", "wb" ))
				return

	# pickle dictionary here
	pickle.dump(dictionary, open( "data/spectrograms/" + data, "wb" ))
	return

def generate_spectrogram_array(inputs):
	"""
	Returns a spectrogram of the given inputs
	:param inputs: array of sample values of a given frame
	"""
	 
	# Stft parameters
	n_fft = 4096
	hop_length = 256
	n_mels = 128
	f_min = 20
	# f_max = 11637
	f_max = 11025
	sample_rate = 22050
	w_length = 1024

	# Librosa melspectrogram
	mels = librosa.feature.melspectrogram(inputs, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=w_length)

	# TODO: experiment with if one of these improves model
	# mels = librosa.power_to_db(mels, ref=np.max)
	# mels = librosa.power_to_db(np.abs(mels)**2, ref=np.max)
	# mels = librosa.power_to_db(np.abs(mels), ref=np.max)

	return mels


def make_spectrogram_image(inputs, filename):
	"""
	Makes an individual spectrogram image and saves it in data/spectrograms/images with the given filename
	:param inputs: array of sample values of a given frame
	:param filename: string, to be the name of the saved spectrogram file
	"""

	f_min = 20
	# f_max = 11637
	f_max = 11025

	mels = generate_spectrogram_array(inputs)

	# Getting rid of edges of figure
	figure = plt.figure(figsize=(500, 600), dpi=1)
	axis = plt.subplot(1, 1, 1)
	plt.axis('off')
	plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')

	S_db = librosa.power_to_db(mels, ref=np.max)

	img = librosa.display.specshow(S_db, fmin=f_min, fmax=f_max)

	extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())

	# Saving figure	
	plt.savefig("data/spectrograms/images/" + filename, bbox_inches=extent, pad_inches=0)

	# Close plots for memory purposes
	plt.clf()
	plt.close()

# unpickle file method
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


# get mix and vocal data method to be called in assignment
def get_data(file_path):
	unpickled_file = unpickle(file_path)
	mix = unpickled_file['mix']
	vocals = unpickled_file['vocals']
	instrumental = unpickled_file['instrumental']

	# slice spectrograms into (,9,2049,1) inputs for model
	mix = slice_spectrogram(mix)
	vocals = slice_spectrogram(vocals)
	instrumental = slice_spectrogram(instrumental)
	
	return mix, vocals, instrumental

def slice_spectrogram(spectrogram):
	"""
	"""

	# from paper - number of continuos frames per vocal harmonic is 9
	length = 9
	hop_length = 256

	slices = []
	for n in range(len(spectrogram)):
		# get array for individual excerpt
		spec = spectrogram[n]

		for x in range(0, hop_length//length):
			s = spec[:, x * length : (x+1) * length]
			slices.append(s)

	slices = np.transpose(slices, (0,2,1))

	# remove excess to fit dimensions
	remainder = len(slices) % 18441
	if remainder != 0:
		slices = slices[:-remainder]

	slices = np.reshape(slices, [-1,length,2049,1])

	return slices


def main():

	# download data from Google Drive
	if sys.argv[1] == "-gdrive":
		print("Using Google Drive to Download Data")
		gdown.download('https://drive.google.com/uc?id=10YE9fmvwVE21Sel8ng7biTC_0ye82R5T&export=download', 'spectrograms.zip', quiet=False)
		with zipfile.ZipFile("spectrograms.zip", 'r') as zip_ref:
			zip_ref.extractall("data")
		os.remove("spectrograms.zip")
		
		print("Data has finished downloading!")
		return

	# locally generate train data 
	elif sys.argv[1] == "-train":
		print("Locally generating train data")
		make_spectrograms("train", False)

	# locally generate test data 
	elif sys.argv[1] == "-test":
		print("Locally generating test data")
		make_spectrograms("test", False)

	# locally generate all data 
	elif sys.argv[1] == "-all":
		print("Locally generating all data")
		make_spectrograms("train", False)
		make_spectrograms("test", False)

	# for testing purposes if we want to make changes to preprocess - only generates 2 new files (1 mix and 1 vocal)
	elif sys.argv[1] == "-1":
		make_spectrograms("train", True)
		return

	else:
		print("A flag must be given when running preprocess.py!")
		return
	
	print("Data has finished generating!")
	return


if __name__ == '__main__':
    main()