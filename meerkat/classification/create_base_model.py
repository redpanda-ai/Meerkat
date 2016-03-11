import os
import time
import sys
from collections import defaultdict
from plumbum import local, NOHUP
from meerkat.classification.tools import create_new_configuration_file, copy_file

#################### USAGE ##########################

# python3 -m meerkat.classification.create_base_model [t7b_folder]
# python3 -m meerkat.classification.create_base_model meerkat/classification/lua/base_model_data/

# t7b_folder is a folder containing training and testing sets for each
# model desired to fold into the base model

#####################################################

def collect_datasets():

	datasets = defaultdict(lambda: {"train": "", "test": ""})

	# Collect training and testing data
	for file in os.listdir(sys.argv[1]):
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
	os.makedirs(output_path + "iteration_stats/", exist_ok=True)
	copy_file("meerkat/classification/lua/main.lua", output_path)
	copy_file("meerkat/classification/lua/data.lua", output_path)
	copy_file("meerkat/classification/lua/model.lua", output_path)
	copy_file("meerkat/classification/lua/train.lua", output_path)
	copy_file("meerkat/classification/lua/test.lua", output_path)

def create_base_model():

	output_path = "base_model_training/"
	datasets = collect_datasets()
	setup_directory(output_path)
	last_trained_model = ""
	models_trained = 0

	# Train each model sequentially and transfer the output model each time
	for model, data in datasets.items():

		# Move data files and create new config
		create_new_configuration_file(None, output_path, output_path + data["train"], output_path + data["test"])
		copy_file("meerkat/classification/lua/base_model_data/" + data["train"], output_path)
		copy_file("meerkat/classification/lua/base_model_data/" + data["test"], output_path)

		# Start training
		if last_trained_model == "":
			with local.cwd(output_path):
				(local["th"]["main.lua"]) & NOHUP
		else:
			with local.cwd(output_path):
				command = (local["th"]["main.lua"]["-transfer"][last_trained_model]) & NOHUP

		# Wait for training to finish one era
		training = True
		while training:
			time.sleep(30)
			for file in os.listdir(output_path):
				if "sequential" in file:
					models_trained += 1
					os.remove(output_path + last_trained_model) if last_trained_model != "" else None
					last_trained_model = "base_model_" + str(models_trained) + ".t7b"
					os.rename(output_path + file, output_path + last_trained_model)
					training = False
					(local["pkill"]["th"])
					break

		# Clean Up Directory
		os.remove(output_path + data["train"])
		os.remove(output_path + data["test"])
		for file in os.listdir(output_path):
			if "main_" in file:
				os.rename(output_path + file, output_path + "iteration_stats/" + model + "_" + str(models_trained) + "_" + file)
				break

if __name__ == "__main__":
	create_base_model()
