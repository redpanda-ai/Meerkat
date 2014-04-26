import csv, sys, json
from meerkat.description_consumer import get_qs_query, get_bool_query
from elasticsearch import Elasticsearch, helpers
from scipy.stats.mstats import zscore
from pprint import pprint

def load_store_numbers(file_name):
	"""Load Store Numbers from provided file"""

	input_file = open(file_name, encoding="utf-8", errors='replace')
	stores = list(csv.DictReader(input_file, delimiter=","))
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

	fields = ["address", "postcode", "name", "locality", "region"]
	search_parts = [store["address"], store["zip_code"][0:5], store["keywords"], store["city"], store["state"]]
	factual_id = ""

	# Generate Query
	bool_search = get_bool_query(size=45)
	should_clauses = bool_search["query"]["bool"]["should"]

	# Multi Field
	for i in range(len(fields)):
		sub_query = get_qs_query(search_parts[i], [fields[i]])
		should_clauses.append(sub_query)

	# Search Index
	results = search_index(bool_search)
	score, top_hit = get_hit(results, 0)

	if score == False:
		return ""

	# Allow User to Verify and Return 
	formatted = [top_hit.get("name", ""), top_hit.get("address", ""), top_hit.get("postcode", ""), top_hit.get("locality", ""), top_hit.get("region", ""),]
	formatted = ", ".join(formatted)
	print("Z-Score: ", score)
	print("Top Result: ", formatted)
	print("Query Sent: ", search_parts)

	# Must Match Keywords
	if not (store["keywords"] in top_hit["name"]):
		return ""	

	if score > 0.95:
		return top_hit["factual_id"]

	return factual_id

def get_hit(search_results, index):

	# Must have results
	if search_results['hits']['total'] == 0:
		return False, False

	hits = search_results['hits']['hits']
	scores = [hit['_score'] for hit in hits]
	z_score = z_score_delta(scores)
	hit = hits[index]

	return z_score, hit["_source"]

def update_merchant(factual_id, store):
	"""Update found merchant with store_number"""

	store_number = store["internal_store_number"]
	while len(store_number) < 4:
		store_number = "0" + store_number
	store_number = "#" + store_number

	body = {"doc" : {"internal_store_number" : store_number}}

	try:
		output_data = es_connection.update(index="factual_index", doc_type="factual_type", id=factual_id, body=body)
	except Exception:
		print("Failed to Update Merchant")

	return output_data["ok"]

def search_index(query):
	"""Searches the merchants index and the merchant mapping"""

	input_data = json.dumps(query, sort_keys=True, indent=4\
	, separators=(',', ': ')).encode('UTF-8')
	output_data = ""

	try:
		output_data = es_connection.search(index="factual_index", body=query)
	except Exception:
		output_data = {"hits":{"total":0}}

	return output_data

def run(stores):
	"""Run the Program"""

	not_found = []
	total = len(stores)

	# Run Search
	for i in range(len(stores)):

		# Find Most Likely Merchant
		store = stores[i]
		factual_id = find_merchant(store)

		# Attempt to Update Document
		if len(factual_id) > 0:
			status = True
			pass
			#status = update_merchant(factual_id, store)
		else:
			print("Did Not Merge Store Number ", store["store_number"], " To Index", "\n")
			not_found.append(store)
			status = False
			continue

		# Save Failed Attempts
		if status == False:
			print("Did Not Merge Store Number ", store["store_number"], " To Index")
			not_found.append(store)
		else:
			print("Successfully Merged Store Number:", store["store_number"], "into Factual Merchant:", factual_id, "\n")

	# Show Success Rate
	misses = len(not_found)
	hits = total - misses
	print("HITS: ", hits)
	print("MISSES: ", misses)
	print("PERCENT MERGED: ", hits / total)

	# Save Not Found
	#save_not_found(not_found)

def save_not_found(not_found):
	"""Save the stores not found in the index"""

	delimiter = ","
	output_file = open("no_results.csv", 'w')
	dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=not_found[0].keys())
	dict_w.writeheader()
	dict_w.writerows(not_found)
	output_file.close()

if __name__ == "__main__":

	cluster_nodes = ["brainstorm0:9200", "brainstorm1:9200", "brainstorm2:9200"
        , "brainstorm3:9200", "brainstorm4:9200", "brainstorm5:9200", "brainstorm6:9200"
        , "brainstorm7:9200", "brainstorm8:9200", "brainstorm9:9200", "brainstorma:9200"
        , "brainstormb:9200"]
	es_connection = Elasticsearch(cluster_nodes, sniff_on_start=True, sniff_on_connection_fail=True, sniffer_timeout=15, sniff_timeout=15)
	file_name = "data/misc/Store Numbers/Dirty/publix_super_markets.csv"
	stores = load_store_numbers(file_name)
	run(stores)
