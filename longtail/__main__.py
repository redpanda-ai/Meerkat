#!/usr/local/bin/python3

from longtail.description_producer import initialize, get_desc_queue, tokenize, load_parameter_key

#Runs the entire program.
PARAMS = initialize()
KEY = load_parameter_key(PARAMS)
print(KEY)
DESC_QUEUE = get_desc_queue(PARAMS)
tokenize(PARAMS, DESC_QUEUE)
