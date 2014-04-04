#!/usr/local/bin/python3
'''Docstring for our __main__.py module'''

from longtail.description_producer import initialize, get_desc_queue\
, tokenize, load_hyperparameters

#Runs the entire program.
PARAMS = initialize()
HYPERPARAMETERS = load_hyperparameters(PARAMS)
DESC_QUEUE, NON_PHYSICAL = get_desc_queue(PARAMS)
tokenize(PARAMS, DESC_QUEUE, HYPERPARAMETERS, NON_PHYSICAL)
