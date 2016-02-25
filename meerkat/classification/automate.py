"""Stop the CNN training process if the threshold reached."""

import os
import sys
import re
import json
import logging
import time
import ctypes

from plumbum import local
import pandas as pd

def getFile():
	"""Get the latest t7b file under current directory."""
	print("Get the latest main_*.t7b file")

	command = local["ls"]["-Falt"] \
			| local["grep"]["main"] \
			| local["head"]["-n"]["1"] \
			| local["awk"]["{print $9}"]

	result = command()
	if result is None or len(result) <= 10: # No main_*.t7b files.
		return None
	else:
		return result[0:-1]

def getCNNStatics(inputFile):
	"""Get the era number and error rate."""
	lualib = ctypes.CDLL("/home/ubuntu/torch/install/lib/libluajit.so", mode=ctypes.RTLD_GLOBAL)

	# Must be imported after previous statement
	import lupa
	from lupa import LuaRuntime

	# Load Runtime and Lua Modules
	lua = LuaRuntime(unpack_returned_tuples=True)
	torch = lua.require('torch')

	template = '''
		function(filename)
			rows = {}
			main = torch.load(filename)
			for i = 0, #main.record - 1 do
				error_rate = main.record[#main.record - i ]["val_error"]
				rows[#main.record -i] = error_rate
			end
			return rows
		end
	'''

	# Load Lua Functions
	get_error = lua.eval(template)
	lua_table = get_error(inputFile)

	error_list = list(lua_table)
	error_vals = list(lua_table.values())
	return dict(zip(error_list, error_vals))

def getTheBestErrorRate(erasDict):
	"""Get the best error rate among different eras"""
	bestErrorRate = 1.0
	bestEraNumber = 1

	df = pd.DataFrame.from_dict(erasDict, orient="index")
	bestErrorRate = df.min().values[0]
	bestEraNumber = df.idxmin().values[0]

	return bestErrorRate, bestEraNumber

def zipDir(file1, file2):
	"""Copy files to Best_CNN_Statics directory and zip it"""
	local["mkdir"]["Best_CNN_Statics"]()
	local["cp"][file1]["Best_CNN_Statics"]()
	local["cp"][file2]["Best_CNN_Statics"]()
	local["tar"]["-zcvf"]["Best_CNN_Statics.tar.gz"]["Best_CNN_Statics"]()

def stopStream():
	"""Stop stream.py when the threshold reached."""
	local["pkill"]["qlua"]

def main_stream():
	"""The main program"""
	fileList = [] # A list to store all the main_*.t7b files.
	threshold = 2 # The highest era number - the era number of the best error rate.

	staticsDict = getCNNStatics(getFile())
	print(staticsDict)
	print(getTheBestErrorRate(staticsDict))
	print(len(staticsDict))

	while True:
		print("Suspend the program for 10 minutes, and wait for a new file.")
		time.sleep(600) # Sleep for 10 minutes.

		latest_t7b = getFile()

		if latest_t7b is not None:
			if len(fileList) == 0 or latest_t7b != fileList[len(fileList) - 1]: # Has new file.
				fileList.append(latest_t7b)

				writeToLuaFile(latest_t7b, "output_statics.lua")
				executeLuaFile("output_statics.lua")
				eras = loadStaticsToMap("staticsJsonFile")
				bestErrorRate, bestEraNumber = getTheBestErrorRate(eras)

				# Stop the training process if threshold meets.
				if len(eras) - bestEraNumber > threshold:
					print("The training process has been stopped.")
					print("The CNN statics files have been zipped in Best_CNN_Statics.")
					zipDir(latest_t7b, "staticsJsonFile")
					stopStream()
					return
				else:
					print("The training process is still on")
					print(bestErrorRate)
					print(bestEraNumber)

			else: # No new file.
				print("No new file detected")
				pass
		else: # latest_t7b is None.
			print("The latest_t7b is None")
			pass

if __name__ == "__main__":
	main_stream()
