#!/usr/bin/python

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with 
ElasticSearch (structured data)."""

import copy, json, sys, re, urllib2
from query_templates import GENERIC_ELASTICSEARCH_QUERY, STOP_WORDS

class InvalidArguments(Exception):
	"""Thrown when invalid arguments are passed in via command line."""
	pass

def begin_parse(input_string):
	"""Creates data structures used the first call into the 
	parse_into_search_tokens function."""

	print "Input: " + input_string
	terms, long_substrings, recursive = [], {}, False	
	parse_into_search_tokens(long_substrings, terms, input_string\
	, recursive)

def display_search_results(hits):
	"""Displays search results."""
	expected_fields = [ "BUSINESSSTANDARDNAME", "HOUSE", "STREET"\
	, "STRTYPE", "CITYNAME", "STATE", "ZIP" ]
	complex_expected_fields = [ "lat", "lon"]

	for item in hits: 
		fields = item['fields']
		#place "blanks" if expected field is null
		for field in expected_fields:
			if field not in fields:
				fields[field] = ""
		for field in complex_expected_fields:
			if "pin.location" not in fields:
				fields["pin.location"] = {}
			if field not in fields["pin.location"]:
				fields["pin.location"][field] = 0.0
				
		#display the search result
		output_format = "{0} {1} {2} {3} {4}, {5} {6} ({7}, {8})"
		print output_format.format(fields["BUSINESSSTANDARDNAME"]\
		, fields["HOUSE"], fields["STREET"], fields["STRTYPE"]\
		, fields["CITYNAME"], fields["STATE"], fields["ZIP"]\
		, fields["pin.location"]["lat"], fields["pin.location"]["lon"])

def display_results():
	"""Displays our tokens, n-grams, and search results."""
	numeric = re.compile("^[0-9/#]+$")

	stop_tokens = []	
	filtered_tokens = []
	numeric_tokens = []
	for token in TOKENS:
		if token in STOP_WORDS:
			stop_tokens.append(token)
		elif numeric.search(token): 
			numeric_tokens.append(token)
		else:
			filtered_tokens.append(string_cleanse(token))

	print "Unigrams are:\n\t" + str(TOKENS)
	print "Unigrams matched to ElasticSearch:\n\t" + str(UNIGRAM_TOKENS)
	print "Of these:"
	print "\t" + str(len(stop_tokens)) + " stop words:      "\
	+ str(stop_tokens)
	print "\t" + str(len(numeric_tokens)) + " numeric words:   "\
	+ str(numeric_tokens)
	print "\t" + str(len(filtered_tokens)) + " worth searching: "\
	+ str(filtered_tokens)

	#show all search terms seperated by spaces		
	query_string = " ".join(filtered_tokens)
	
	
	print "Search Attempt using n-grams, n = 1"
	print "You can also try Google:\n\t" + str(query_string)
	get_query_string_search_results(query_string)

	#I think these results should be refined, as they sometimes over-score
	#results that should be less relevant.
	print "Search Attempt using n-grams, n >= 1"
	print "WARNING: These results may over-score irrelevant n-grams."
	print "\tIn the future, we can revise these results."
	n_gram_tokens = get_n_gram_tokens(filtered_tokens)
	matched_n_gram_tokens = search_n_gram_tokens(n_gram_tokens)
	get_composite_search_results(matched_n_gram_tokens)

def extract_longest_substring(long_substrings, longest_substring):
	"""Extracts the longest substring from our input, returning left over fragments
	We now use the longest_substring, to find the following strings:
	1.  "original_term": the original string containing the longest_substring
	2.  "pre": substring preceding the longest_substring
	3.  "post" : substring following the longest_substring"""
	
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

def get_composite_search_results(matched_n_gram_tokens):
	"""Obtains search results for a query composed of multiple
	sub-queries.  At some point I may add 'query_string' as an
	additional parameter."""
	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 10, 0
	
	sub_query = {}
	sub_query["match"] = {}
	sub_query["match"]["_all"] = {}
	sub_query["match"]["_all"]["query"] = "__term"
	sub_query["match"]["_all"]["type"] = "phrase"
	sub_query["match"]["_all"]["boost"] = 1.2

	for term in matched_n_gram_tokens:
		my_sub_query = copy.deepcopy(sub_query)
		my_sub_query["match"]["_all"]["query"] = term	
		my_new_query["query"]["bool"]["should"].append(my_sub_query)

	output = search_index(my_new_query)	
	hits = output['hits']['hits']
	display_search_results(hits)

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
	input_string = json.dumps(input_as_object)
	url = "http://brainstorm8:9200/merchants/merchant/_search"
	request = urllib2.Request(url, input_string)
	response = urllib2.urlopen(request)
	METRICS["query_count"] += 1
	output = json.loads(response.read())
	return output

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

	print "Matched the following n-grams to our merchants index:"
	for key in reversed(sorted(n_gram_tokens.iterkeys())):
		for term in n_gram_tokens[key]:
			sub_query["match"]["_all"]["query"] = term
			my_new_query["query"]["bool"]["should"].append(sub_query)
			hit_count = search_index(my_new_query)['hits']['total']
			del my_new_query["query"]["bool"]["should"][0]
			if hit_count > 0:
				print str(key) + "-gram : " + term + " (" + str(hit_count) + ")"
				matched_n_gram_tokens.append(term)
	return matched_n_gram_tokens
	

def get_query_string_search_results(my_query_string):
	"""Runs an ElasticSearch query over all unigram tokens at once
	to find a set of scored results."""
	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 10, 0

	sub_query = {}
	sub_query["query_string"] = {}
	sub_query["query_string"]["query"] = "__term"
	my_new_query["query"]["bool"]["should"].append(sub_query)
	sub_query["query_string"]["query"] = "".join(my_query_string)
	
	output = search_index(my_new_query)	
	hits = output['hits']['hits']
	display_search_results(hits)

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

	#This block finds the longest substring match in our index and adds it to our 
	#dictionary of search terms.	
	longest_substring = long_substrings[sorted\
	(long_substrings.iterkeys())[-1]].keys()[0]
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

	for key in reversed(sorted(substrings.iterkeys())):
		for term in substrings[key]:
			hit_count = 0
			if len(term) > 1:
				sub_query["query_string"]["query"] = str(term)
				hit_count = search_index(my_new_query)['hits']['total']
			if hit_count > 0:
				return term

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [ "\[", "\]", "'", "\{", "\}", '"', "/"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)	

def usage():
	"""Shows the user which parameters to send into the program."""
	print "Usage:\n\t<quoted_transaction_description_string>"

#INPUT_STRING = "MEL'S DRIVE-IN #2 SAN FRANCISCOCA 24492153337286434101508"

STILL_BREAKABLE = 2
UNIGRAM_TOKENS, TOKENS = [], []
METRICS = { "query_count" : 0 }
INPUT_STRING = initialize()
begin_parse(INPUT_STRING)
display_results()
print "The total number of ElasticSearch queries issued: " + \
str(METRICS["query_count"])
