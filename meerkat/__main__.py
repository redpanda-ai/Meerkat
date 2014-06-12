#!/usr/local/bin/python3.3

"""This file is the file first run when 
Meerkat is executed as a package"""

from meerkat.description_producer import mode_switch, initialize

#Runs the entire program.
params = initialize()
mode_switch(params)

