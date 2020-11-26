import os
import tensorflow as tf
import numpy as np
import math
from convolution import conv2d
from matplotlib import pyplot as plt
import librosa


###
# Model
###
class Model(tf.keras.Model):
	def __init__(self):
		"""
		Constructor of the convolutional neural network model.
		"""
		super(Model, self).__init__()

		self.conv_1 = tf.keras.layers.Conv2D(32, [ , ], padding='same', input_shape=[None, 513, 25, 1], activation='leaky_relu')

		self.conv_2 = tf.keras.layers.Conv2D(16, [ , ], padding='same', input_shape=[None, 513, 25, 32], activation='leaky_relu')
		# self.conv_2_maxpool = tf.keras.layers.MaxPooling2D(pool_size=[ , ], strides=None, padding="valid", data_format=None)
		# self.dropout_2 = tf.keras.layers.Dropout(rate=, noise_shape=None, seed=None)

		self.conv_3 = tf.keras.layers.Conv2D(64, [ , ], padding='same', input_shape=[None, 171, 8, 16], activation='leaky_relu')

		self.conv_4 = tf.keras.layers.Conv2D(16, [ , ], padding='same', input_shape=[None, 171, 8, 64], activation='leaky_relu')
		# self.conv_2_maxpool = tf.keras.layers.MaxPooling2D(pool_size=[ , ], strides=None, padding="valid", data_format=None)
		# self.dropout_2 = tf.keras.layers.Dropout(rate=, noise_shape=None, seed=None)

		self.dense_1 = tf.keras.layers.Dense()
		# self.dropout_2 = tf.keras.layers.Dropout(rate=, noise_shape=None, seed=None)

		self.dense_2 = tf.keras.layers.Dense()


	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: spectrograms of full mixes
		:returns: 
		"""
		conv_1_out = self.conv_1(inputs)

		conv_2_out = self.conv_2(conv_1_out)
		conv_2_out = tf.nn.max_pool(conv_2_out, [ , ], [ , , , ], '')
		conv_2_out = tf.nn.dropout(conv_2_out, rate=)

		conv_3_out = self.conv_3(conv_2_out)

		conv_4_out = self.conv_4(conv_3_out)
		conv_4_out = tf.nn.max_pool(conv_4_out, [ , ], [ , , , ], '')
		conv_4_out = tf.nn.dropout(conv_4_out, rate=)

		conv_4_out_flattened = tf.flatten(conv_4_out)

		dense_1_out = self.dense_1(conv_4_out_flattened)
		dense_1_out = tf.nn.dropout(dense_1_out, rate=)

		dense_2_out = self.dense_2(dense_1_out)

		return dense_2_out

	def loss(self, predictions, actual):
		"""

		:param predictions: 
		:param actual:
		:return: the total loss over the batch
		"""

		pass

	def accuracy(self, predictions, actual):
		"""

		:param predictions:
		:param actual:
		:return: the total accuracy over the batch
		"""

		pass


###
# Training & Testing
###
def train(model, train_inputs, train_labels):
	"""

	:param train_inputs: array representations of the ground truth full mixes for training
	:param train_labels: array representations of the ground truth vocal stems for training
	:return:
	"""
	train_mixes = make_spectrogram(train_inputs)
	train_vocals = make_spectrogram(train_labels)


	pass


def test(model, test_inputs, test_labels):
	"""

	:param test_inputs: array representations of the ground truth full mixes for testing
	:param test_labels: array representations of the ground truth vocal stems for testing
	:return: total test accuracy
	"""
	test_mixes = make_spectrogram(test_inputs)
	test_vocals = make_spectrogram(test_labels)


	pass


def main():
    """

    :return: None
    """
    train_mixes, train_vocals, test_mixes, test_vocals = get_data('data/')

    model = Model()

    # train(model, train_mixes, train_vocals)

    # print(test(model, test_mixes, test_vocals))

    return


if __name__ == '__main__':
    main()