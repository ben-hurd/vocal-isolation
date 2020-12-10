import os
import tensorflow as tf
import numpy as np
import math
from matplotlib import pyplot as plt
import librosa
from preprocess import get_data
from postprocess import spectrogram_to_audio


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
		self.loss_list_test = []

		# TODO: experiment with replacing "12"s with "3"s and seeing if it helps. Also could try 'same' padding
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

	# # shuffle inputs
	# rand_ind = tf.random.shuffle(range(num_examples))
	# mix = tf.gather(mix, rand_ind)
	# vocal = tf.gather(vocal, rand_ind)
	# instrumental = tf.gather(instrumental, rand_ind)

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

		pred = model(mix_batch)
		loss = model.loss(pred, IBM)

		model.loss_list_test.append(loss)

	return


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

	train_mix, train_vocals, train_instrumental = get_data("data/spectrograms/train")
	test_mix, test_vocals, test_instrumental = get_data("data/spectrograms/test")

	# # TRIAL VALUES:
	# train_mix = tf.constant(tf.random.truncated_normal([5000,9,2049,1],stddev=0.1), dtype=tf.float32)
	# train_vocals = tf.constant(tf.random.truncated_normal([5000,9,2049,1],stddev=0.1))
	# train_instrumental = tf.constant(tf.random.truncated_normal([5000,9,2049,1],stddev=0.1),dtype=tf.float32)

	# test_mix = tf.constant(tf.random.truncated_normal([5000,9,2049,1],stddev=0.1), dtype=tf.float32)
	# test_vocals = tf.constant(tf.random.truncated_normal([5000,9,2049,1],stddev=0.1))
	# test_instrumental = tf.constant(tf.random.truncated_normal([5000,9,2049,1],stddev=0.1),dtype=tf.float32)

	# # example of how to go from spectrogram -> audio
	# spectrogram_to_audio(train_mix[5000],"data/train-mix-1.wav")
	# spectrogram_to_audio(train_vocals[5000],"data/train-vocals-1.wav")
	# spectrogram_to_audio(train_instrumental[5000],"data/train-instrumental-1.wav")

	model = Model()

	for i in range(5):

		train(model, train_mix, train_vocals, train_instrumental)

		test(model, test_mix, test_vocals, test_instrumental)

	print("Train loss per batch:")
	visualize_loss(model.loss_list_train)

	print("Test loss per batch")
	visualize_loss(model.loss_list_test)

	return


if __name__ == '__main__':
    main()