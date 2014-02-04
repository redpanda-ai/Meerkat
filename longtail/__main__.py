#!/usr/local/bin/python3

from longtail.description_producer import initialize, get_desc_queue, tokenize

#Runs the entire program.
PARAMS = initialize()
DESC_QUEUE = get_desc_queue(PARAMS)
tokenize(PARAMS, DESC_QUEUE)
