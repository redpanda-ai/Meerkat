import os
import logging
from plumbum import local

def getFile():
	logging.info("Get the latest main_*.t7b file")
	command = local["ls"]["-Falt"] | local["grep"]["main"] | local["head"]["-n"]["1"]
	latestFile = command()
	print(latestFile)

def main_stream():
	getFile()

if __name__ == "__main__":
	main_stream()
