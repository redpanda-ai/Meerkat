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

def build_cnn():
	"""Build CNN"""

	x = tf.placeholder(tf.float32, shape=[128, 1, 69, 123])
	w = tf.Variable(tf.random_normal([1, 69, 123, 256], name="W"))
	tf.nn.conv2d(x, w, [1,1,1,1], padding="SAME")

if __name__ == "__main__":
	build_cnn()