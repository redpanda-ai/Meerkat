from plumbum import local
from plumbum.cmd import sleep, sudo
from requests.exceptions import ConnectionError
from multiprocessing.pool import ThreadPool
import json
import requests
import unittest
from nose_parameterized import parameterized

def check_status():
	"""Get a status code (e.g. 200) from the web service"""
	r = requests.get("https://localhost/status/index.html", verify=False)
	status = r.status_code
	print(r.content)
	r.connection.close()
	return (status)

def get_trans_text(path):
	"""Get the json string of one_ledger.json"""
	transFile = open(path, 'rb')
	transText = transFile.read()
	transFile.close()
	return transText

def classify_one(self, transaction, max_retries=10, sleep_interval=2):
	"""Send a single transaction to the web service for classification"""
	sleep_interval = 2
	count = 1
	while count <= max_retries:
		try:
			sleep(sleep_interval)
			r_post = requests.post(
				"https://localhost/meerkat/v1.7",
				data=transaction,
				verify=False)

			r_post.connection.close()
			self.assertTrue(r_post.status_code == 200)
			break
		except ConnectionError:
			count += 1

	return r_post.content

class WebServiceTest(unittest.TestCase):
	"""Our UnitTest class."""

	@classmethod
	def setUpClass(cls):
		sudo[local["supervisorctl"]["restart"]["meerkat"]]()

	@classmethod
	def tearDownClass(cls):
		sudo[local["supervisorctl"]["stop"]["meerkat"]]()

	def test_web_service_status(self):
		"""Test starts meerkat, checks status of meerkat, and stops meerkat web service"""
		count = 1
		sleep_interval = 2
		max_retries = 10
		# Wait for sleep_interval seconds before trying up to
		# max_retries times
		while count <= max_retries:
			try:
				sleep(sleep_interval)
				status = check_status()
				self.assertTrue(status == 200)
				return
			except ConnectionError:
				count += 1
		self.assertTrue(0 == 1, "Failed to connect to Meerkat service")
		return

	def test_web_service_races(self):
		"""Test starts meerkat, runs 100 classifications, and stops meerkat"""
		samples = 100
		pool = ThreadPool(samples)
		transText = get_trans_text('./web_service_tester/one_ledger.json')
		tasks = []
		for i in range(samples):
			tasks.append(pool.apply_async(
				     classify_one,
				     (self, transText)))
		classified = []
		for task in tasks:
			classified.append(sorted(json.loads(
					  task.get().decode("utf-8")
					  )))
		for i in range(1, samples):
			self.assertEqual(
					 classified[0],
					 classified[i],
					 "Two results were not equal\n{}\n\n{}".format(classified[0], classified[i]))
		return

	@parameterized.expand([
		('./data/input/input.json'),
		('./data/input/web_service_input.json'),
		('./data/input/issue_512.json')
	])
	def test_web_service_representative(self, path):
		"""Test starts meerkat, runs representative transactions, and stop meerkat"""
		transText = get_trans_text(path)
		task = classify_one(self, transText)
		return

if __name__ == "__main__":
	unittest.main()
