#!/usr/local/bin/python3

"""Train a CNN using tensorFlow

Created on Mar 14, 2016
@author: Matthew Sevrens
@author: Tina Wu
"""

#################### USAGE #######################

# python3 -m meerkat.classification.tf_CNN

##################################################

import tensorflow as tf
import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHA_DICT = {a : i for i, a in enumerate(ALPHABET)}

def string_to_tensor(str, l):
	"""Convert transaction to tensor format"""

	s = str.lower()[0:l]
 	t = np.zeros((len(ALPHABET), l), dtype=np.float32)
 	for i, c in reversed(list(enumerate(s))):
 		if c in ALPHABET:
 			t[ALPHA_DICT[c]][len(s) - i - 1] = 1
 	return t

def build_cnn():
	"""Build CNN"""

	graph = tf.Graph()
	num_labels = 10
	batch_size = 128
	doc_length = 123
	alphabet_length = len(ALPHABET)

	# Create Graph
	with graph.as_default():

		x = tf.placeholder(tf.float32, shape=[batch_size, 1, alphabet_length, doc_length])
		y = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		w = tf.Variable(tf.random_normal([1, alphabet_length, doc_length, 256], name="W"))
		tf.nn.conv2d(x, w, [1,1,1,1], padding="SAME")

def run_session(graph):
	"""Run Session"""

	# Train Network
	epochs = 5000
	eras = 10

	with tf.Session(graph=graph) as session:

		tf.initialize_all_variables().run()
		num_eras = epochs * eras

		for step in range(num_eras):

			if (step % epochs == 0):

				print("Save details")

if __name__ == "__main__":
	build_cnn()