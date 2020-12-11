import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from matplotlib import pyplot as plt
import librosa
from preprocess import get_data
from postprocess import spectrogram_to_audio
import sys
import musdb
from preprocess import slice_spectrogram


###
# Model
###
class Model(tf.keras.Model):
	def __init__(self):
		"""
		Constructor of the convolutional neural network model.
		"""
		super(Model, self).__init__()

		self.batch_size = 100
		self.optimizer = tf.keras.optimizers.Adam()
		self.loss_list_train = []
		self.acc_list = []

		self.conv_1 = tf.keras.layers.Conv2D(32, [3,12], strides=1, padding='valid', activation='relu')
		self.conv_2 = tf.keras.layers.Conv2D(16, [3,12], strides=1, padding='valid', activation='relu')

		self.conv_2_maxpool = tf.keras.layers.MaxPooling2D(pool_size=[1,12])

		self.conv_3 = tf.keras.layers.Conv2D(64, [3,12], strides=1, padding='valid', activation='relu')
		self.conv_4 = tf.keras.layers.Conv2D(32, [3,12], strides=1, padding='valid', activation='relu')

		self.conv_4_maxpool = tf.keras.layers.MaxPooling2D(pool_size=[1,12])

		self.dropout_1 = tf.keras.layers.Dropout(rate=0.5)
		self.flatten_1 = tf.keras.layers.Flatten()
		self.dense_1 = tf.keras.layers.Dense(2048, activation='relu', dtype=tf.float32)

		self.dropout_2 = tf.keras.layers.Dropout(rate=0.5)
		self.dense_2 = tf.keras.layers.Dense(512, activation='relu', dtype=tf.float32)
		self.dense_3_out = tf.keras.layers.Dense(18441, activation='sigmoid', dtype=tf.float32)


	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: spectrograms of full mixes
		:return: 
		"""
		conv_1_out = self.conv_1(inputs)

		conv_2_out = self.conv_2(conv_1_out)
		conv_2_out = self.conv_2_maxpool(conv_2_out)

		conv_3_out = self.conv_3(conv_2_out)

		conv_4_out = self.conv_4(conv_3_out)
		conv_4_out = self.conv_4_maxpool(conv_4_out)
		conv_4_out = self.dropout_1(conv_4_out)

		conv_4_out = self.flatten_1(conv_4_out)

		dense_1_out = self.dense_1(conv_4_out)
		dense_1_out = self.dropout_2(dense_1_out)

		dense_2_out = self.dense_2(dense_1_out)


		dense_3_out = self.dense_3_out(dense_2_out)
		dense_3_out = tf.reshape(dense_3_out, [-1,9,2049])

		return dense_3_out

	def loss(self, predictions, actual):
		"""
		Calculates the loss over a given batch
		:param predictions: 
		:param actual:
		:return: the total loss over the batch
		"""

		loss = -tf.math.log(predictions)*actual - tf.math.log(1-predictions)*(1-actual)
		loss = tf.reduce_mean(loss)

		return loss

	def accuracy(self, predictions, actual):
		correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(actual, 1))
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

###
# Training & Testing
###
def train(model, mix, vocal, instrumental):
	"""
	Trains the model to learn a binary mask for vocal isolation
	:param train_inputs: array representations of the ground truth full mixes for training
	:param train_labels: array representations of the ground truth vocal stems for training
	:return:
	"""

	num_examples = len(mix)

	# shuffle inputs
	rand_ind = tf.random.shuffle(range(num_examples))
	mix = tf.gather(mix, rand_ind)
	vocal = tf.gather(vocal, rand_ind)
	instrumental = tf.gather(instrumental, rand_ind)

	# remove excess to be divisible by batch size
	batch_remainder = num_examples % model.batch_size
	if batch_remainder != 0:
		mix = mix[:-batch_remainder]
		vocal = vocal[:-batch_remainder]
		instrumental = instrumental[:-batch_remainder]

	# batch processing
	for i in range(0, num_examples, model.batch_size):
		mix_batch = mix[i : i + model.batch_size]
		vocal_batch = vocal[i : i + model.batch_size]
		instrumental_batch = instrumental[i : i + model.batch_size]

		# ideal binary mask for vocal-dominated pixels
		IBM = tf.squeeze(tf.cast(tf.greater(vocal_batch, instrumental_batch), tf.float32))

		with tf.GradientTape() as tape:
			pred = model(mix_batch)
			loss = model.loss(pred, IBM)

			model.loss_list_train.append(loss)

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return


def test(model, mix, vocal, instrumental):
	"""
	Tests the model on vocal isolation
	:param test_inputs: array representations of the ground truth full mixes for testing
	:param test_labels: array representations of the ground truth vocal stems for testing
	:return: None
	"""
	
	num_examples = len(mix)

	batch_size = 100

	# remove excess to be divisible by batch size
	batch_remainder = num_examples % batch_size
	if batch_remainder != 0:
		mix = mix[:-batch_remainder]
		vocal = vocal[:-batch_remainder]
		instrumental = instrumental[:-batch_remainder]

	acc_list = []
	# batch processing
	x = 0
	for i in range(0, num_examples - batch_remainder, batch_size):
		x += 1
		mix_batch = mix[i : i + batch_size]
		vocal_batch = vocal[i : i + batch_size]
		instrumental_batch = instrumental[i : i + batch_size]

		# ideal binary mask for vocal-dominated pixels
		IBM = tf.squeeze(tf.cast(tf.greater(vocal_batch, instrumental_batch), tf.float32))

		pred = model(mix_batch)
		pred = tf.cast(tf.greater(pred, 0.3), tf.float32)

		correct_predictions = tf.cast(tf.equal(pred,IBM), tf.float32)

		acc = tf.reduce_sum(correct_predictions).numpy() / tf.size(pred).numpy()

		acc_list.append(acc)

	return sum(acc_list[:-1]) / x


# (copied from hw2)
def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def main():
	"""
	:return: None
	"""
	
	## run file with -train flag to train the model for 5 epochs and print its accuracy on the training set 
	## model will be saved at the end to "trained_model"
	if sys.argv[1] == "-train":

		train_mix, train_vocals, train_instrumental = get_data("data/spectrograms/train")
		test_mix, test_vocals, test_instrumental = get_data("data/spectrograms/test")

		model = Model()

		for i in range(5):

			train(model, train_mix, train_vocals, train_instrumental)
			print("End of Epoch:", i+1)

		print("Training loss:")
		visualize_loss(model.loss_list_train)

		acc = test(model, test_mix, test_vocals, test_instrumental)
		print("Model Accuracy is:", acc)
		model.save("trained_model")
		return

	## run file with -savedWeights flag to used the saved model and print its accuracy on the training set
	## 2nd flag that references a .wav mix can be added and an isolated-vocals.wav file will be produced
	## that is our model's attempt at isolating the vocals from that song (likely won't completely finish due to errors in our reshaping)
	elif sys.argv[1] == "-savedWeights":

		# load model
		model = keras.models.load_model('trained_model', compile=False)

		# test saved model
		test_mix, test_vocals, test_instrumental = get_data("data/spectrograms/test")

		# print accuracy 
		acc = test(model, test_mix, test_vocals, test_instrumental)
		print("Model Accuracy is:", acc)

		# if 2nd flag exists, treat it as a .wav file of a full mix, and save model's output to isolated_vocals.wav
		if sys.argv[2]:
			y, sr = librosa.load(sys.argv[2], sr=None)
			original_sr = sr
			target_sr = 22050
			mix_data = librosa.resample(librosa.to_mono(y), orig_sr=original_sr, target_sr=target_sr, res_type='kaiser_best', fix=True, scale=False)
			
			len_frame = target_sr*3
			num_frames = int(len(mix_data)/len_frame)

			totalMels = []

			for frame in range(num_frames):
				n_fft = 4096
				hop_length = 256
				n_mels = 128
				sample_rate = 22050
				w_length = 1024

				# Librosa melspectrogram
				mels = librosa.feature.melspectrogram(mix_data[frame * len_frame : frame * len_frame + len_frame], sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, win_length=w_length)
				totalMels.append(mels)

			x = len(totalMels)
			y = len(totalMels[0])
			z = len(totalMels[0][0])

			Input = slice_spectrogram(totalMels)

			IBM = model(Input)
			IBM = tf.squeeze(tf.cast(tf.greater(IBM, 0.0), tf.float32))

			vocals = tf.multiply(IBM,tf.cast(tf.squeeze(Input),tf.float32))
			vocals = tf.squeeze(tf.reshape(vocals,[-1,1]))


			difference = x * y * z - vocals.shape[0]
			vocals = tf.cast(vocals.numpy(), tf.float32)
			pad = tf.cast(tf.zeros(difference), tf.float32)

			vocals = tf.concat([vocals,pad],0)

			vocals = tf.reshape(vocals, [y,x*z])
			# likely will freeze here due to the way we reshape :(
			spectrogram_to_audio(vocals.numpy(), "isolated-vocals.wav")

		return 

	return


if __name__ == '__main__':
    main()
