"""Stop the CNN training process if the threshold reached."""

import time
import json
import os
import sys
import ctypes
from plumbum import local

from .tools import(getFile, getCNNStatics, getTheBestErrorRate, zipDir, stopStream)

def main_stream():
	"""The main program"""
	fileList = [] # A list to store all the main_*.t7b files.
	threshold = 2 # The highest era number - the era number of the best error rate.

	while True:
		print("Suspend the program for 10 minutes, and wait for a new file.")
		time.sleep(10) # Sleep for 10 minutes.

		latest_t7b = getFile(sys.argv[1])

		if latest_t7b is not None:
			if len(fileList) == 0 or latest_t7b != fileList[len(fileList) - 1]: # Has new file.
				fileList.append(latest_t7b)
				local["cp"][sys.argv[1] + "/" + latest_t7b]["."]()

				staticsDict = getCNNStatics(latest_t7b)
				bestErrorRate, bestEraNumber = getTheBestErrorRate(staticsDict)

				# Stop the training process if threshold meets.
				if len(staticsDict) - bestEraNumber > threshold:
					print("The training process has been stopped.")
					print("The CNN statics files have been zipped in Best_CNN_Statics.")
					json.dump(staticsDict, open("text.txt", "w"))
					zipDir(fileList[bestEraNumber - 1], "text.txt")
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
