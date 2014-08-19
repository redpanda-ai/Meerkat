#!/usr/local/bin/python3.3

"""This file is the file first run when 
Meerkat is executed as a package"""

import os, re

from meerkat.producer import mode_switch, initialize, validate_params
from meerkat.various_tools import purge

# Run the Meerkat Classifier

try:

	params = initialize()
	validate_params(params)
	mode_switch(params)

except (KeyboardInterrupt, SystemExit) as e:

	input_dir = params["input"]["split"]["processing_location"]
	output_dir = params["output"]["file"]["processing_location"]
	purge(input_dir, "output*")
	purge(output_dir, "output*")