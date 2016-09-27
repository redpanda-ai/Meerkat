"""
This module will download all RNN CNN sws models from s3 to local instance
@author: Oscar Pan

USAGE
python3 -m meerrkat.auto_load_all
"""

import logging

import meerkat.longtail.rnn_auto_load as load_rnn
import meerkat.longtail.sws_auto_load as load_sws
import meerkat.classification.auto_load as load_cnn

def load_all():
	"""Load all models"""
	logging.basicConfig(level=logging.INFO)
	logging.info("Start loading RNNs")
	load_rnn.auto_load()
	logging.info("Start loading SWSs")
	load_sws.sws_auto_load()
	logging.info("Start loading CNNs")
	load_cnn.main_program()

if __name__ == "__main__":
	load_all()
