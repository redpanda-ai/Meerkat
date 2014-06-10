#!/usr/local/bin/python3
'''Docstring for our __main__.py module'''

from meerkat.description_producer import mode_switch, initialize

#Runs the entire program.
params = initialize()
mode_switch(params)

