import requests
import sys
import logging
import time
import json

from plumbum import local
from plumbum.cmd import sudo
from requests.exceptions import ConnectionError

def get_trans_text(path):
	"""Get the json string of json file"""
	transFile = open(path, 'rb')
	transaction_text = transFile.read()
	transFile.close()
	return transaction_text

def classify_one(transaction, max_retries=20, sleep_interval=2):
	"""Send a single transaction to the web service for classification"""
	count = 1
	while count <= max_retries:
		try:
			time.sleep(sleep_interval)
			r_post = requests.post(
				"https://localhost/meerkat/v2.2",
				data=transaction,
				verify=False)

			r_post.connection.close()
			if r_post.status_code == 200:
				break
		except ConnectionError:
			count += 1
	return r_post.content

def startup_helper(transaction, max_retries=100, sleep_interval=1):
	"""Send a single transaction to the web service for classification"""
	count = 1
	while count <= max_retries:
		try:
			time.sleep(sleep_interval)
			r_post = requests.post(
				"https://localhost/meerkat/v2.2",
				data=transaction,
				verify=False)
			r_post.connection.close()
			if r_post.status_code == 200:
				logging.warning("Web service startup complete")
				break
		except ConnectionError:
			count += 1
			logging.warning("Web service startup time: {0}".format(count * sleep_interval))
	if count >= max_retries:
		logging.critical("Web service failed to start, aborting.".format(count * sleep_interval))
		sys.exit()
	return r_post.content

def main_program():
	logging.basicConfig(level=logging.INFO)
	sudo[local["supervisorctl"]["restart"]["meerkat"]]()
	path = sys.argv[1]
	transaction_text = get_trans_text(path)
	_ = startup_helper(transaction_text)
	logging.warning("Meerkat fully online.")
	classified = classify_one(transaction_text)
	result = json.loads(classified.decode("utf-8"))
	logging.info(json.dumps(result, indent=4))

if __name__ == "__main__":
	main_program()
