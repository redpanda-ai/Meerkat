import os
import sys
import re
import json
import logging
from plumbum import local
import pandas as pd

def getFile():
	"""Get the latest t7b file under current directory"""
	logging.info("Get the latest main_*.t7b file")
	command = local["ls"]["-Falt"] \
			| local["grep"]["main"] \
			| local["head"]["-n"]["1"] \
			| local["awk"]["{print $9}"]

	latestFile = command()[0:-1]
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
	# A list of dictionaries.
	eras = []

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

def main_stream():
	"""The main program"""
	latest_t7b = getFile()
	writeToLuaFile(latest_t7b, "output_statics.lua")
	executeLuaFile("output_statics.lua")

	eras = loadStaticsToMap("staticsJsonFile")
	print(len(eras))
	print(eras[0])

if __name__ == "__main__":
	main_stream()
