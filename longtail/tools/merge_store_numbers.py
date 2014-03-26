import csv
from longtail.description_consumer import get_qs_query
from elasticsearch import Elasticsearch, helpers
from scipy.stats.mstats import zscore

def load_store_numbers(file_name):
	"""Load Store Numbers from provided file"""

	input_file = open(file_name, encoding="utf-8", errors='replace')
	stores = list(csv.DictReader(input_file))
	input_file.close()

	return stores

def z_score_delta(scores):
	"""Find the Z-Score Delta"""

	if len(scores) < 2:
		return None

	z_scores = zscore(scores)
	first_score, second_score = z_scores[0:2]
	z_score_delta = round(first_score - second_score, 3)
	
	return z_score_delta

def find_merchant(store):
	"""Match document with store number to factual document"""

	address = store["address"]
	zip_code = store["zip_code"]

def update_merchant(store_number):
	"""Update found merchant with store_number"""	

if __name__ == "__main__":

	keywords = ["mcdonalds", "mcdonald's"]
	file_name = "data/misc/Store Numbers/Clean/mcdonalds_store_numbers.pipe"
	stores = load_store_numbers(file_name)

	print(len(stores))