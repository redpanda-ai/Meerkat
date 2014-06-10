#!/usr/local/bin/python3
'''Docstring for our __main__.py module'''

from meerkat.description_producer import process_bucket, initialize

#Runs the entire program.
params = initialize()
process_bucket(params)

