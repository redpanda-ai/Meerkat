import os
import sys
import argparse

import tensorflow as tf

from .tools import get_cost_list
from meerkat.classification.data_handler import download_data
from meerkat.classification.distributed_train import inference, placeholder_inputs

def parse_arguments(args):
	parser = argparse.ArgumentParser("get_meta")
	help_text = {
		"model_type" : "Pick a valid Classifier type",
		"bank_or_card": "Is the model being trained on card or bank transactions?",
		"credit_or_debit": "Is the model for credit or debit transactions?"
	}
	choices = {
		"model_type": ["subtype", "merchant", "category"],
		"bank_or_card": ["bank", "card"]
	}
	# Required arugments
	parser.add_argument("model_type", help=help_text["model_type"], choices=choices["model_type"])
	parser.add_argument("bank_or_card", help=help_text["bank_or_card"], choices=choices["bank_or_card"])

	parser.add_argument("--credit_or_debit", default='', help=help_text["credit_or_debit"])

	args = parser.parse_args(args)

	if (args.model_type == 'subtype' or args.model_type == 'category') and args.credit_or_debit == '':
		raise Exception('For subtype data you need to declare debit or credit.')

	return args

def get_meta(config, base):

	with tf.Session() as sess:
		cost_list = get_cost_list(config)
		trans_placeholder, labels_placeholder = placeholder_inputs(config)
		network, model, bn_assigns = inference(trans_placeholder, config)
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		learning_rate = tf.train.exponential_decay(config["base_rate"], global_step, 15000, 0.5, staircase=True)

		# Calculate Loss and Optimize
		weighted_labels = cost_list * labels_placeholder
		loss = tf.neg(tf.reduce_mean(tf.reduce_sum(network * weighted_labels, 1)), name="loss")
		optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step, name="optimizer")
		tf.scalar_summary('loss', loss)

		bn_updates = tf.group(*bn_assigns)
		with tf.control_dependencies([optimizer]):
			bn_applier = tf.group(bn_updates, name="bn_applier")

		saver = tf.train.Saver()
		init_op = tf.initialize_all_variables()
		sess.run(init_op)
		saver.save(sess, base + 'train.ckpt')
	os.remove(base + 'checkpoint')
	os.remove(base + 'train.ckpt')
	return base + 'train.ckpt.meta'

if __name__ == "__main__":
	args = parse_arguments(sys.argv[1:])
	config, _, _ = download_data(args.model_type, args.bank_or_card, args.credit_or_debit)
	get_meta(config, "meerkat/classification/models/")
