import os
import sys
import re
import json
import logging
from plumbum import local

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
	"""Load the training statics to a hashMap"""
	key_val_re = re.compile("\s*([a-zA-z]+)\s:\s*(.+)")
	index_re = re.compile("\s*([0-9]+)\s:\s*(.+)")
	clean_re = re.compile(r'\x1b[^m]*m')
	eras = []
	my_era = {}
	with open(filename, encoding="utf-8") as input_file:
		for line in input_file:
			#print(line)
			if key_val_re.match(line):
				m = key_val_re.search(line)
				my_era[m.group(1)] = clean_re.sub('', m.group(2))
				#print('"{0}" : "{1}"'.format(m.group(1), m.group(2)))
			elif index_re.match(line):
				if len(my_era) != 0:
					eras.append(my_era)
				my_era = {}
	import pandas as pd


	my_dfs = []
	for c in range(len(eras)):
		print("Era {0}".format(c+1))
		print(eras[c])
	#print(result)

	sys.exit()
#		return json.loads(input_file.read())
	#inputfile = open(filename, encoding='utf-8')
	#hashmap = json.load(inputfile.read())
	#return hashmap

def main_stream():
	"""The main program"""
	latest_t7b = getFile()
	writeToLuaFile(latest_t7b, "output_statics.lua")
	executeLuaFile("output_statics.lua")

	hashmap = loadStaticsToMap("staticsJsonFile")

if __name__ == "__main__":
	main_stream()
