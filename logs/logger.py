"""
Adapted from here:
http://www.blog.pythonlibrary.org/2014/02/11/python-how-to-create-rotating-logs/

This file just simply records the time in rotating logs. The logs are cycled
once every day, with a maximum of 7 logs existing at any point in time.
"""

import logging
import time
import os

from logging.handlers import TimedRotatingFileHandler

#----------------------------------------------------------------------
def create_timed_rotating_log(path):
	""""""
	logger = logging.getLogger("Rotating Log")
	logger.setLevel(logging.INFO)

	# create a handler to rotate 7 logs, with each one generated every
	# 10 seconds
	handler = TimedRotatingFileHandler(path,
									   when = "s",
									   interval = 10,
									   backupCount = 7)
	logger.addHandler(handler)
	
	# add dummy info to the logs, just for demonstration purposes
	while True:
	 	print(os.listdir("."))
	 	logger.info(time.time())
	 	time.sleep(3)
 
#----------------------------------------------------------------------
if __name__ == "__main__":
	log_file = "timed_test.log"
	create_timed_rotating_log(log_file)
