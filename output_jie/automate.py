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

	latestFile = command()
	return latestFile

def main_stream():
	latest_t7b = getFile()
	print(latest_t7b)

if __name__ == "__main__":
	main_stream()
