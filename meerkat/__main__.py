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

	if params["input"].get("S3", "") != "":
		input_dir = params["input"]["S3"]["src_local_path"]
		output_dir = params["input"]["S3"]["dst_local_path"]
		purge(input_dir, "output*")
		purge(output_dir, "output*")