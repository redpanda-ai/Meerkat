from plumbum import ProcessExecutionError
from plumbum import local, BG
from plumbum.cmd import sudo, kill, grep, python3, sleep
from requests.exceptions import ConnectionError

import requests
import unittest

def web_service_is_online():
	"""Tests to see if the web service is already online."""
	try:
		no = sudo["ps"]["-ef"] | grep["python"] | \
		grep["root"]| grep ["meerkat.web_service"]
		web_service_pid = no().split()[1]
		return True, web_service_pid
	except ProcessExecutionError:
		return False, None

def start_web_service():
	"""Starts the web service as a background process."""
	start = sudo[python3["-m"]["meerkat.web_service"]]
	with local.cwd("/home/ubuntu/git/Meerkat"):
		(start) & BG

def stop_linux_process(my_pid):
	"""Stops any linux process with the provided process id (my_pid) """
	try:
		sudo[kill[my_pid]]()
	except ProcessExecutionError:
		print("Unable to kill, aborting")
		sys.exit()

def check_status():
	"""Get a status code (e.g. 200) from the web service"""
	r = requests.get("https://localhost/status/index.html", verify=False)
	status = r.status_code
	r.connection.close()
	return (status)

class WebServiceTest(unittest.TestCase):
	"""Our UnitTest class."""
	
	def setUp(self):
		online, web_service_pid = web_service_is_online()
		if online:
			stop_linux_process(web_service_pid)

	def tearDown(self):
		online, web_service_pid = web_service_is_online()
		if online:
			stop_linux_process(web_service_pid)

	def test_web_service_status(self):
		"""Test starts, checks status of, and stops meerkat web service"""
		start_web_service()
		count, sleep_interval, max_retries = 1, 2, 10
		#Wait for sleep_interval seconds before trying up to
		#max_retries times
		while count <= max_retries:
			try:
				sleep(sleep_interval)
				status = check_status()
				self.assertTrue(status == 200)
				return
			except ConnectionError:
				count += 1
		return

if __name__ == "__main__":
	unittest.main()
