#!/usr/bin/python

import copy, json, os, sys, re, urllib, urllib2
from subprocess import Popen, PIPE
from query_templates import GENERIC_ELASTICSEARCH_QUERY

class InvalidArguments(Exception):
	pass

STILL_BREAKABLE = 2
UNIGRAM_SIZE = 1

bad_characters = [ "\[", "\]", "'", "\{", "\}", '"', "/"]
x = "|".join(bad_characters)
cleanse_pattern = re.compile(x)

unigram_tokens = []
stop_words = ["CHECK", "CARD", "CHECKCARD", "PAYPOINT", "PURCHASE", "LLC" ]
expected_fields = [ "BUSINESSSTANDARDNAME", "HOUSE", "STREET", "STRTYPE", \
"CITYNAME", "STATE", "ZIP" ]
complex_expected_fields = [ "lat", "lon"]

#This function displays our output
def display_results():
	numeric = re.compile("^[0-9/#]+$")

	stop_tokens = []	
	filtered_tokens = []
	numeric_tokens = []
	for token in tokens:
		if token in stop_words:
			stop_tokens.append(token)
		elif numeric.search(token): 
			numeric_tokens.append(token)
		else:
			filtered_tokens.append(string_cleanse(token))

	print "Unigrams are:\n\t" + str(tokens)
	print "Unigrams matched to ElasticSearch:\n\t" + str(unigram_tokens)
	print "Of these:"
	print "\t" + str(len(stop_tokens)) +    " stop words:      " + str(stop_tokens)
	print "\t" + str(len(numeric_tokens)) + " numeric words:   " + str(numeric_tokens)
	print "\t" + str(len(filtered_tokens)) + " worth searching: " + str(filtered_tokens)
	#show all search terms seperated by spaces		
	query_string = " ".join(filtered_tokens)
	n_gram_tokens = get_n_gram_tokens(filtered_tokens)
	matched_n_gram_tokens = search_n_gram_tokens(n_gram_tokens)

	
	print "Search Attempt using n-grams = 1"
	print "You can also try Google:\n\t" + str(query_string)
	get_query_string_search_results(query_string)

	#FIXME these results actually suck, and aren't worth getting yet
	#print "Search Attempt using n-grams > 1"
	#print "These results currently are not especially good or relevant"
	#get_composite_search_results(matched_n_gram_tokens,query_string)

def get_composite_search_results(matched_n_gram_tokens,query_string):
	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 10,0
	
	sub_query = {}
	sub_query["match"] = {}
	sub_query["match"]["_all"] = {}
	sub_query["match"]["_all"]["query"] = "__term"
	sub_query["match"]["_all"]["type"] = "phrase"
	sub_query["match"]["_all"]["boost"] = 1.2

	for term in matched_n_gram_tokens:
		s = copy.deepcopy(sub_query)
		s["match"]["_all"]["query"] = term	
		my_new_query["query"]["bool"]["should"].append(s)

	output = search_index(my_new_query)	
	hits = output['hits']['hits']
	display_hits(hits)

def get_n_gram_tokens(list_of_tokens):
	n_gram_tokens = {}
	x = len(list_of_tokens)
	for j in range(x):
		for i in range(x-j):
			maybe = list_of_tokens[i:x-j]
			y = len(maybe)
			if y > UNIGRAM_SIZE:
				if y not in n_gram_tokens:
					n_gram_tokens[y] = []
				n_gram_tokens[y].append(" ".join(maybe))

	return n_gram_tokens

def search_n_gram_tokens(n_gram_tokens):
	matched_n_gram_tokens = []

	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 0,0

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
	
#Extracts the longest substring from our input, returning left over fragments
def extract_longest_substring(longest_substring):
	#We now use the longest substring, to find the following strings:
	#1.  "original_term": the string that contained the longest substring
	#2.  "pre": everything from the original term before the longest_substring
	#3.  "post" : everything from the original term after the longest_substring
	
	l = len(longest_substring)	
	original_term = longest_substrings[l][longest_substring]
	b = original_term.find(longest_substring)
	c = b + len(longest_substring)
	pre, post = original_term[:b], original_term[c:]

	#delete the current "longest substring" from our dictionary
	del longest_substrings[l][longest_substring]
	if len(longest_substrings[l]) == 0:
		del longest_substrings[l]

	return original_term, pre, post

#Searches the merchants index and the merchant mapping
def search_index(input_as_object):
	input = json.dumps(input_as_object)
	url = "http://brainstorm8:9200/merchants/merchant/_search"
	request = urllib2.Request(url, input)
	response = urllib2.urlopen(request)
	output = json.loads(response.read())
	return output

#Runs an ElasticSearch query to find the results
def get_query_string_search_results(unigram_tokens):
	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 10,0

	sub_query = {}
	sub_query["query_string"] = {}
	sub_query["query_string"]["query"] = "__term"
	my_new_query["query"]["bool"]["should"].append(sub_query)
	sub_query["query_string"]["query"] = "".join(unigram_tokens)
	
	output = search_index(my_new_query)	
	hits = output['hits']['hits']
	display_hits(hits)

def display_hits(hits):
	for item in hits: 
		f = item['fields']
		#place "blanks" if expected field is null
		for xf in expected_fields:
			if xf not in f:
				f[xf] = ""
		for cf in complex_expected_fields:
			if "pin.location" not in f:
				f["pin.location"] = {}
			if cf not in f["pin.location"]:
				f["pin.location"][cf] = 0.0
				
		#display the search result
		print "{0} {1} {2} {3} {4}, {5} {6} ({7}, {8})".format(f["BUSINESSSTANDARDNAME"]\
		,f["HOUSE"], f["STREET"], f["STRTYPE"],f["CITYNAME"],f["STATE"],f["ZIP"]\
		,f["pin.location"]["lat"], f["pin.location"]["lon"])

#Validates the command line arguments
def initialize():
	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments("Incorrect number of arguments")
	return sys.argv[1]

#Recursively Finds all substrings for a term
def powerset(term,substrings):
	l = len(term)
	if l not in substrings:
		substrings[l] = {}
	if term not in substrings[l]:
		substrings[l][term] = ""
	if l <= STILL_BREAKABLE:
		return
	if term in stop_words:
		return
	else:
		powerset(term[0:-1],substrings)
		powerset(term[1:],substrings)

#Looks for the longest substring in our merchants index that 
#can actually be found.
def search_substrings(substrings):
	my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
	my_new_query["size"], my_new_query["from"] = 0,0

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

#Recursively attempts to parse an unstructured transaction description into tokens
def parse_description_into_search_tokens(longest_substrings, terms, input, recursive):
	global tokens
	
	#This loop breaks up the input looking for new search terms
	#that match portions of our ElasticSearch index
	if len(input) >= STILL_BREAKABLE:
		new_terms = input.split()
		for term in new_terms:
			term = string_cleanse(term)
			if not recursive:
				tokens.append(term)
			substrings = {}
			powerset(term,substrings)
			local_longest_substring = search_substrings(substrings)
			if local_longest_substring is not None:
				l = len(local_longest_substring)
				if l not in longest_substrings:
					longest_substrings[l] = {}
				if local_longest_substring not in longest_substrings[l]:
					longest_substrings[l][local_longest_substring] = term
		terms.extend(new_terms)

	#This check allows us to exit if no substrings are found
	if len(longest_substrings) == 0:
		return	

	#This block finds the longest substring match in our index and adds it to our 
	#dictionary of search terms.	
	longest_substring = longest_substrings[sorted(longest_substrings.iterkeys())[-1]].keys()[0]
	unigram_tokens.append(longest_substring)

	original_term, pre, post = extract_longest_substring(longest_substring)
	tokens = rebuild_tokens(original_term,longest_substring,pre,post)

	parse_description_into_search_tokens(longest_substrings, terms, pre + " " + post, True)

#Rebuild our complete list of tokens following a substring extraction
def rebuild_tokens(original_term, longest_substring, pre, post):
	rebuilt_tokens = []
	for i in range(len(tokens)):
		if tokens[i] != original_term:	
			rebuilt_tokens.append(tokens[i])
		else:
			if pre != "":
				rebuilt_tokens.append(pre)
			rebuilt_tokens.append(longest_substring)
			if post != "":
				rebuilt_tokens.append(post)
	return rebuilt_tokens

def string_cleanse(s):
	return re.sub(cleanse_pattern,"",s)	

def usage():
	print "Usage:\n\t<quoted_transaction_description_string>"

#input = "MEL'S DRIVE-IN #2 SAN FRANCISCOCA 24492153337286434101508"

input = initialize()

longest_substrings, terms, tokens = {}, [], []
print "Input: " + input
recursive = False
parse_description_into_search_tokens(longest_substrings, terms, input, recursive)
display_results()

