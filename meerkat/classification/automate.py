"""Stop the CNN training process if the threshold reached.
@author: Jie Zhang
"""

############################## USAGE ########################
"""
python3 -m meerkat.classification.automate directory
"""
#############################################################

import time
import json
import os
import sys
import ctypes
import logging

from plumbum import local
from .tools import(get_new_maint7b, getCNNStatics, getTheBestErrorRate,
		zipDir, stopStream)

def main_stream(directory):
	"""The main program"""
	# store initial files in directory
	fileList = os.listdir(directory)
	threshold = 2 # The highest era number - the era number of the best error rate.

	seconds = 30
	while True:
		print("Suspend the program for {0} minute, and wait for a new file."\
			.format(seconds/60.0))
		time.sleep(seconds)

		latest_t7b = get_new_maint7b(directory, fileList)

		if latest_t7b is not None:
			local["cp"][directory + "/" + latest_t7b]["."]()

			staticsDict = getCNNStatics(latest_t7b)
			best_error, best_era = getTheBestErrorRate(staticsDict)

			# Stop the training process if threshold meets.
			if len(staticsDict) - best_era > threshold:
				print("The training process has been stopped.")
				# print("The CNN statics files have been zipped in "
					# "Best_CNN_Statics.")
				json.dump(staticsDict, open(directory + "/all_error_rates", "w"))
				# zipDir(fileList[best_era - 1], "all_error_rates")
				stopStream()
				return directory + latest_t7b.replace('main','sequential')
			else:
				print("The training process is still on")
				print('Best error rate is {0} at era {1}.'.format(best_error, best_era))

		else: # No new file.
			print("No new maint7b detected")

if __name__ == "__main__":
	logging.info("This module should be running only when a "
		"model training is in progress.")
	_ = main_stream(sys.argv[1])
