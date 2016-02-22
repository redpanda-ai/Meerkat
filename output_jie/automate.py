import os
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

def main_stream():
	latest_t7b = getFile()
	print(latest_t7b)

	writeToLuaFile(latest_t7b, "output_statics.lua")

if __name__ == "__main__":
	main_stream()
