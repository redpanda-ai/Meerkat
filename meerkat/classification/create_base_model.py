import os
import time
from collections import defaultdict
from meerkat.classification.tools import stopStream, execute_main_lua, create_new_configuration_file
from meerkat.classification.tools import copy_file

#################### USAGE ##########################

# python3 -m meerkat.classification.create_base_model

#####################################################

def collect_datasets():

	datasets = defaultdict(lambda: {"train": "", "test": ""})

	# Collect training and testing data
	for file in os.listdir("meerkat/classification/lua/base_model_data/"):
		if file.endswith(".t7b"):
			file_trunc = file.replace(".test.poor.t7b", "")
			file_trunc = file_trunc.replace(".train.poor.t7b", "")
			if "train" in file:
				datasets[file_trunc]["train"] = file
			elif "test" in file:
				datasets[file_trunc]["test"] = file

	return datasets

def setup_directory(output_path):

	os.makedirs(output_path, exist_ok=True)
	copy_file("meerkat/classification/lua/main.lua", output_path)
	copy_file("meerkat/classification/lua/data.lua", output_path)
	copy_file("meerkat/classification/lua/model.lua", output_path)
	copy_file("meerkat/classification/lua/train.lua", output_path)
	copy_file("meerkat/classification/lua/test.lua", output_path)

def create_base_model():

	datasets = collect_datasets()
	setup_directory("base_model_training/")

	# Train each model sequentially and transfer the output model each time
	for model in datasets.keys():

		# Move data files and create new config
		create_new_configuration_file(None, output_path, model["train"], model["test"])
		copy_file("meerkat/classification/lua/base_model_data/" + model["train"], output_path)
		copy_file("meerkat/classification/lua/base_model_data/" + model["test"], output_path)

if __name__ == "__main__":
	create_base_model()