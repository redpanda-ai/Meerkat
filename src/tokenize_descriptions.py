#!/usr/bin/python

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with
ElasticSearch (structured data)."""

import copy, json, logging, sys, re, urllib.request
from query_templates import GENERIC_ELASTICSEARCH_QUERY, STOP_WORDS\
, get_match_query, get_qs_query, RESULT_FIELDS
from various_tools import string_cleanse
from custom_exceptions import InvalidArguments, UnsupportedQueryType
from scipy.stats.mstats import zscore

def begin_parse(input_string):
	"""Creates data structures used the first call into the
	parse_into_search_tokens function."""
	print( "Input String ",input_string)
	terms, long_substrings, recursive = [], {}, False
	parse_into_search_tokens(long_substrings, terms, input_string\
	, recursive)

def generate_complete_boolean_query(unigrams, address, phone_numbers):
	"""Constructs a complete boolean query based upon:
	1.  Unigrams (query_string)
	2.  Addresses (match)
	3.  Phone Number (match)"""
	#Add dynamic output based upon a dictionary of token types
	#See comment above
	search_components = []
	print( "Search components are:")
	print( "\tUnigrams: ", "'", unigrams, "'")
	#search_components.append((unigrams, "qs_query", ["_all"]))
	search_components.append((unigrams, "qs_query", ["_all^1"\
	, "BUSINESSSTANDARDNAME^2"], 1))
	if address is not None:
		print( "\tMatching 'Address': " , "'" , address , "'")
		search_components.append((address, "match_query"\
		,["composite.address^1"], 10))
	if len(phone_numbers) != 0:
		for phone_num in phone_numbers:
			print( "\tMatching 'Phone': " , "'" , phone_num , "'")
			search_components.append((phone_num, "match_query"\
			, ["composite.phone^1"], 8))

	my_obj = get_boolean_search_object(search_components)
	print( my_obj)
	my_results = search_index(my_obj)
	print( "This system required " , str(METRICS["query_count"])\
		, " individual searches.")
	display_search_results(my_results)

def display_results():
	"""Displays our tokens, n-grams, and search results."""
	phone_re = re.compile("^[0-9/#]{10}$")
	numeric = re.compile("^[0-9/#]+$")

	stop_tokens = []
	filtered_tokens = []
	numeric_tokens = []
	addresses = []
	phone_numbers = []
	for token in TOKENS:
		if token in STOP_WORDS:
			stop_tokens.append(token)
		elif phone_re.search(token):
			phone_numbers.append(string_cleanse(token))
		elif numeric.search(token):
			numeric_tokens.append(token)
		else:
			filtered_tokens.append(string_cleanse(token))

	#Add dynamic output based upon a dictionary of token types
	#e.g. Unigram, Composite, Numeric, Stop...
	logging.info( "Unigrams are:\n\t" + str(TOKENS))
	logging.info( "Unigrams matched to ElasticSearch:\n\t" + str(UNIGRAM_TOKENS))
	logging.info( "Of these:")
	logging.info( "\t" + str(len(stop_tokens)) + " stop words:      "\
				+ str(stop_tokens))
	logging.info( "\t" + str(len(phone_numbers)) + " phone_numbers:   "\
		+ str(phone_numbers))
	logging.info( "\t" + str(len(numeric_tokens)) + " numeric words:   "\
		+ str(numeric_tokens))
	logging.info( "\t" + str(len(filtered_tokens)) + " unigrams: "\
		+ str(filtered_tokens))

	count, matching_address = get_matching_address()
	if count > 0:
		addresses.append(matching_address)
	logging.info( "\t" + str(len(addresses)) + " addresses: " + str(addresses))

	#show all search terms separated by spaces
	query_string = " ".join(filtered_tokens)

	generate_complete_boolean_query(query_string, matching_address\
	, phone_numbers)
	#Not finished, do a tail recursive search for longest matching
	#n-gram on a per composite-feature basis.
	#n_gram_tokens = get_n_gram_tokens(filtered_tokens)
	#n_gram_tokens = get_n_gram_tokens(TOKENS)

	##print n_gram_tokens
	##Not currently used
	##matched_n_gram_tokens = search_n_gram_tokens(n_gram_tokens)
	##print "ALL"

def display_z_score_delta(scores):
	"""Display the Z-score delta between the first and second scores."""
	z_scores = zscore(scores)
	first_score, second_score = z_scores[0:2]
	z_score_delta = round(first_score - second_score, 3)
	print( "Z-Score delta: [" , str(z_score_delta) , "]")
	quality = "Non"
	if z_score_delta <= 1:
		quality = "Low-grade"
	elif z_score_delta <= 2:
		quality = "Mid-grade"
	else:
		quality = "High-grade"
	print( "Top Score Quality: " , quality)

def display_search_results(search_results):
	"""Displays search results."""
	hits = search_results['hits']['hits']
	scores, results = [], []
	for hit in hits:
		hit_fields, score = hit['fields'], hit['_score']
		scores.append(score)
		field_order = RESULT_FIELDS
		fields_in_hit = [field for field in hit_fields]
		ordered_hit_fields = []
		for ordinal in field_order:
			if ordinal in fields_in_hit:
				my_field = str(hit_fields[ordinal])
				ordered_hit_fields.append(my_field)
		results.append(\
		"[" + str(round(score,3)) + "] " + " ".join(ordered_hit_fields))
	display_z_score_delta(scores)
	for result in results:
		print( result)

def get_matching_address():
	"""Sadly, this function is temporary.  I plan to go to a more
	generic approach that exhaustively works with all n-grams
	against all composite features."""
	numeric = re.compile("^[0-9]+$")
	address_candidates = {}
	n_gram = []
	for token in TOKENS:
		if numeric.search(token):
			n_gram = []
			n_gram.append(token)
		else:
			n_gram.append(token)
			current_length = len(n_gram)
			if current_length > 1:
				if current_length not in address_candidates:
					address_candidates[current_length] = []
					my_stuff = \
					address_candidates[current_length]
					my_stuff.append(" ".join(n_gram))
	for key in reversed(sorted(address_candidates.keys())):
		candidate_list = address_candidates[key]
		count, term = get_composite_search_count(candidate_list\
		,"composite.address")
		if count > 0:
			return count, term
	return 0, None

def extract_longest_substring(long_substrings, longest_substring):
	"""Extracts the longest substring from our input, returning left
	over fragments to find the following strings:
	1.  "original_term": the original string containing "longest_substring"
	2.  "pre": substring preceding "longest_substring"
	3.  "post" : substring following "longest_substring" """

	ls_len = len(longest_substring)
	original_term = long_substrings[ls_len][longest_substring]
	start_index = original_term.find(longest_substring)
	end_index = start_index + ls_len
	pre, post = original_term[:start_index], original_term[end_index:]

	#delete the current "longest substring" from our dictionary
	del long_substrings[ls_len][longest_substring]
	if len(long_substrings[ls_len]) == 0:
		del long_substrings[ls_len]

	return original_term, pre, post

def get_boolean_search_object(search_components):
	"""Builds an object for a "bool" search."""
	bool_search = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	bool_search["size"], bool_search["from"] = 10, 0

	for item in search_components:
		my_subquery = None
		term, query_type, feature_list, boost = item[0:4]
		if query_type == "qs_query":
			my_subquery = get_qs_query(term, feature_list, boost)
		elif query_type == "match_query":
			for feature in feature_list:
				my_subquery = get_match_query(term, feature, boost)
		else:
			raise UnsupportedQueryType("There is no support"\
			+ " for a query of type: " + query_type )
		bool_search["query"]["bool"]["should"].append(my_subquery)
	return bool_search

def get_composite_search_count(list_of_ngrams, feature_name):
	"""Obtains search results for a query composed of multiple
	sub-queries.  At some point I may add 'query_string' as an
	additional parameter."""
	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 0, 0

	sub_query = {}
	sub_query["match"] = {}
	sub_query["match"][feature_name] = {}
	sub_query["match"][feature_name]["query"] = "__term"
	sub_query["match"][feature_name]["type"] = "phrase"
	sub_query["match"][feature_name]["boost"] = 1.2

	for term in list_of_ngrams:
		my_sub_query = copy.deepcopy(sub_query)
		my_sub_query["match"][feature_name]["query"] = term
		my_new_query["query"]["bool"]["should"].append(my_sub_query)
		total = search_index(my_new_query)['hits']['total']
		if total == 0:
			my_new_query["query"]["bool"]["should"] = []
		else:
			return total, term
	return 0, None

def get_n_gram_tokens(list_of_tokens):
	"""Generates a list of n-grams where n >= 2."""
	unigram_size = 1
	n_gram_tokens = {}
	end_index = len(list_of_tokens)
	for end_offset in range(end_index):
		for start_index in range(end_index-end_offset):
			n_gram_size = end_index - end_offset - start_index
			if n_gram_size > unigram_size:
				if n_gram_size not in n_gram_tokens:
					n_gram_tokens[n_gram_size] = []
				new_n_gram = \
				list_of_tokens[start_index:end_index-end_offset]
				n_gram_tokens[n_gram_size].append(" ".join(new_n_gram))
	return n_gram_tokens

def search_index(input_as_object):
	"""Searches the merchants index and the merchant mapping"""
	input_data = json.dumps(input_as_object).encode('UTF-8')
	url = "http://brainstorm8:9200/"
	path = "merchants/merchant/_search"
	req = urllib.request.Request(url=url+path,data=input_data)
	output_data = urllib.request.urlopen(req).read().decode('UTF-8')
	METRICS["query_count"] += 1
	output_string = json.loads(output_data)
	return output_string

def search_n_gram_tokens(n_gram_tokens):
	"""Creates a boolean elasticsearch composed of multiple
	sub-queries.  Each sub-query is itself a 'phrase' query
	built of n-grams where n >= 2."""
	matched_n_gram_tokens = []

	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 0, 0

	sub_query = {}
	sub_query["match"] = {}
	sub_query["match"]["_all"] = {}
	sub_query["match"]["_all"]["query"] = "__term"
	sub_query["match"]["_all"]["type"] = "phrase"

	print( "Matched the following n-grams to our merchants index:")
	for key in reversed(sorted(n_gram_tokens.keys())):
		for term in n_gram_tokens[key]:
			sub_query["match"]["_all"]["query"] = term
			my_new_query["query"]["bool"]["should"].append(sub_query)
			hit_count = search_index(my_new_query)['hits']['total']
			del my_new_query["query"]["bool"]["should"][0]
			if hit_count > 0:
				print( str(key) , "-gram : " , term , " (" , str(hit_count) , ")")
				matched_n_gram_tokens.append(term)
	return matched_n_gram_tokens

def initialize():
	"""Validates the command line arguments."""
	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments("Incorrect number of arguments")
	return sys.argv[1]

def parse_into_search_tokens(long_substrings, terms, input_string, recursive):
	"""Recursively attempts to parse an unstructured transaction
	description into TOKENS."""
	global TOKENS
	#This loop breaks up the input string looking for new search terms
	#that match portions of our ElasticSearch index
	if len(input_string) >= STILL_BREAKABLE:
		new_terms = input_string.split()
		for term in new_terms:
			term = string_cleanse(term)
			if not recursive:
				TOKENS.append(term)
			substrings = {}
			powerset(term, substrings)
			big_substring = search_substrings(substrings)
			if big_substring is not None:
				bs_length = len(big_substring)
				if bs_length not in long_substrings:
					long_substrings[bs_length] = {}
				if big_substring not in long_substrings[bs_length]:
					long_substrings[bs_length][big_substring] = term
		terms.extend(new_terms)

	#This check allows us to exit if no substrings are found
	if len(long_substrings) == 0:
		return

	#This block finds the longest substring match in our index and adds
	#it to our dictionary of search terms
	longest_substring = list(long_substrings[sorted\
	(long_substrings.keys())[-1]].keys())[0]
	UNIGRAM_TOKENS.append(longest_substring)

	original_term, pre, post = extract_longest_substring(\
	long_substrings, longest_substring)
	TOKENS = rebuild_tokens(original_term, longest_substring, pre, post)

	parse_into_search_tokens(long_substrings, terms, pre + " " + post, True)

def powerset(term, substrings):
	"""Recursively discover all substrings for a term."""
	term_length = len(term)
	if term_length not in substrings:
		substrings[term_length] = {}
	if term not in substrings[term_length]:
		substrings[term_length][term] = ""
	if term_length <= STILL_BREAKABLE:
		return
	if term in STOP_WORDS:
		return
	else:
		powerset(term[0:-1], substrings)
		powerset(term[1:], substrings)

def rebuild_tokens(original_term, longest_substring, pre, post):
	"""Rebuilds our complete list of tokens following a substring
	extraction."""
	rebuilt_tokens = []
	for i in range(len(TOKENS)):
		if TOKENS[i] != original_term:
			rebuilt_tokens.append(TOKENS[i])
		else:
			if pre != "":
				rebuilt_tokens.append(pre)
			rebuilt_tokens.append(longest_substring)
			if post != "":
				rebuilt_tokens.append(post)
	return rebuilt_tokens

def search_substrings(substrings):
	"""Looks for the longest substring in our merchants index that
	#can actually be found."""
	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 0, 0

	sub_query = {}
	sub_query["query_string"] = {}
	sub_query["query_string"]["query"] = "__term"
	my_new_query["query"]["bool"]["should"].append(sub_query)

	for key in reversed(sorted(substrings.keys())):
		for term in substrings[key]:
			hit_count = 0
			if len(term) > 1:
				sub_query["query_string"]["query"] = str(term)
				hit_count = search_index(my_new_query)['hits']['total']
			if hit_count > 0:
				return term

def usage():
	"""Shows the user which parameters to send into the program."""
	print( "Usage:\n\t<quoted_transaction_description_string>")

#INPUT_STRING = "MEL'S DRIVE-IN #2 SAN FRANCISCOCA 24492153337286434101508"

STILL_BREAKABLE = 2
UNIGRAM_TOKENS, TOKENS = [], []
METRICS = { "query_count" : 0 }
INPUT_STRING = initialize()
begin_parse(INPUT_STRING)
display_results()

