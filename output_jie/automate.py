import os
import sys
import re
import json
import logging
import time

from plumbum import local
import pandas as pd

def getFile():
	"""Get the latest t7b file under current directory"""
	logging.info("Get the latest main_*.t7b file")
	command = local["ls"]["-Falt"] \
			| local["grep"]["main_"] \
			| local["head"]["-n"]["1"] \
			| local["awk"]["{print $9}"]

	result = command()

	'''No main_*.t7b files'''
	if result is None or len(result) == 0:
		return None

	latestFile = result[0:-1]
	return latestFile

def writeToLuaFile(inputFileName, outputLuaFile):
	"""Write three lines of code to the outputLuaFile"""
	fpointer = open(outputLuaFile, "w")
	fpointer.write("main = torch.load('" + inputFileName + "')\n" + \
					"records = main.record\n" + \
					"print(records)\n")

def executeLuaFile(luaFile):
	"""Execute the input lua file"""
	command = local["th"][luaFile] > "staticsJsonFile"
	command()

def loadStaticsToMap(filename):
	"""Load the training statics to a list of dictionaries"""
	key_val_re = re.compile("\s*([a-zA-z]+)\s:\s*(.+)")
	index_re = re.compile("\s*([0-9]+)\s:\s*(.+)")
	clean_re = re.compile(r'\x1b[^m]*m')

	eras = [] # A list of dictionaries.
	my_era = {}
	with open(filename, encoding="utf-8") as input_file:
		for line in input_file:
			if key_val_re.match(line):
				m = key_val_re.search(line)
				my_era[m.group(1)] = clean_re.sub('', m.group(2))
			elif index_re.match(line):
				if len(my_era) != 0:
					eras.append(my_era)
				my_era = {}
	return eras

def getTheBestErrorRate(eras):
	"""Get the best error rate among different eras"""
	bestErrorRate = 1.0
	bestEra = 0

	for i in range(len(eras)):
		if float(eras[i]["val_error"]) < bestErrorRate:
			bestErrorRate = float(eras[i]["val_error"])
			bestEraNumber = i + 1
	return bestErrorRate, bestEraNumber

def main_stream():
	"""The main program"""
	fileList = [] # A list to store all the main_*.t7b files.
	threshold = 2 # The highest era number - the era number of the best error rate.

	while True:
		print("Suspend the program for 1 minutes, and wait for a new file.")
		time.sleep(20) # Sleep for 1 minutes.

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
					print(fileList)
					return
				else:
					print("The training process is still on")
					print(bestErrorRate)
					print(bestEraNumber)
					return

			else: # No new file.
				print("No new file detected")
				pass
		else: # latest_t7b is None.
			print("The latest_t7b is None")
			pass

if __name__ == "__main__":
	main_stream()
