#!/usr/bin/python

import json, os, sys, re, urllib, urllib2
from subprocess import Popen, PIPE
from query_templates import unpack_attributes, unpack_json_id, standard_query

class InvalidArguments(Exception):
	pass

bad_characters = [ "\[", "\]", "'", "\{", "\}", '"', "/"]
x = "|".join(bad_characters)
cleanse_pattern = re.compile(x)

elasticsearch_tokens = []
stop_words = ["CHECK", "CARD", "CHECKCARD", "PAYPOINT", "PURCHASE", "LLC" ]
expected_fields = [ "BUSINESSSTANDARDNAME", "HOUSE", "STREET", "STRTYPE", \
"CITYNAME", "STATE", "ZIP" ]
complex_expected_fields = [ "lat", "lon"]

#This function displays our output
def display_results():
	numeric = re.compile("^[0-9/#]+$")

	search_tokens = []
	stop_tokens = []	
	filtered_tokens = []
	numeric_tokens = []
	for token in tokens:
		if token in elasticsearch_tokens:
			search_tokens.append(token)
		if token in stop_words:
			stop_tokens.append(token)
		elif numeric.search(token): 
			numeric_tokens.append(token)
		else:
			filtered_tokens.append(string_cleanse(token))

	print "Unigrams are:\n\t" + str(tokens)
	print "Unigrams matched to ElasticSearch: " + str(elasticsearch_tokens)
	print "Of these:"
	print "\t" + str(len(stop_tokens)) +    " stop words:      " + str(stop_tokens)
	print "\t" + str(len(numeric_tokens)) + " numeric words:   " + str(numeric_tokens)
	print "\t" + str(len(filtered_tokens)) + " worth searching: " + str(filtered_tokens)
	#show all search terms seperated by spaces		
	show_terms = " ".join(filtered_tokens)
	ngrams = get_ngrams(filtered_tokens)
	display_ngrams(ngrams)
	#TODO: search the ngrams with a "match" query using "bool" to combine them with the "term query"
	print "Elasticsearching the following"
	print "You can also try Google:\n\t" + str(show_terms)
	get_elasticsearch_results(show_terms)

def get_ngrams(list_of_tokens):
	ngrams = {}
	x = len(list_of_tokens)
	for j in range(x):
		for i in range(x-j):
			maybe = list_of_tokens[i:x-j]
			y = len(maybe)
			if y > 1:
				if y not in ngrams:
					ngrams[y] = []
				ngrams[y].append(" ".join(maybe))
	return ngrams

def display_ngrams(ngrams):
	query_key = "query.match._all"
	query_values = [ ("query",'"__term"'), ("type",'"phrase"') ]

	print "Matched the following n-grams to our merchants index:"
	for key in reversed(sorted(ngrams.iterkeys())):
		for term in ngrams[key]:
			attributes = [ ("size",0) ]
			hit_count = search_index(term,standard_query,attributes,query_key,query_values)['hits']['total']
			if hit_count > 0:
				print str(key) + "-gram : " + term + " (" + str(hit_count) + ")"
	
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
def search_index(term,action,attributes,query_key,query_values):
	url = "http://brainstorm8:9200/merchants/merchant/_search"

	input = action.replace("__attributes",(unpack_attributes(attributes)))
	input = input.replace("__query", unpack_json_id(query_key, query_values)[1:-1])
	input = input.replace("__term",urllib.quote_plus(term))

	request = urllib2.Request(url, input)
	response = urllib2.urlopen(request)
	output = json.loads(response.read())
	return output

#Runs an ElasticSearch query to find the results
def get_elasticsearch_results(elasticsearch_tokens):
	query_key = "query.query_string"
	query_values = [ ("query",'"__term"') ]
	attributes = [ ("from",0), ("size",10) ]
	output = search_index(elasticsearch_tokens,standard_query,attributes,query_key,query_values)
	hits = output['hits']['hits']
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
	if l <= 2:
		return
	if term in stop_words:
		return
	else:
		powerset(term[0:l-1],substrings)
		powerset(term[1:l],substrings)

#Looks for the longest substring in our merchants index that 
#can actually be found.
def search_substrings(substrings):
	query_key = "query.query_string"
	query_values = [ ("query",'"__term"') ]
	attributes = [ ("size",0) ]
	for key in reversed(sorted(substrings.iterkeys())):
		for term in substrings[key]:
			hit_count = 0
			if len(term) > 1:
				hit_count = search_index(term,standard_query,attributes,query_key,query_values)['hits']['total']
			if hit_count > 0:
				return term

#Recursively attempts to parse an unstructured transaction description into tokens
def parse_into_tokens(longest_substrings, terms, input, recursive):
	global tokens
	
	#This loop breaks up the input looking for new search terms
	#that match portions of our ElasticSearch index
	if len(input) >= 2:
		new_terms = input.split()
		for term in new_terms:
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
	#unsorted dictionary of search terms.	
	longest_substring = longest_substrings[sorted(longest_substrings.iterkeys())[-1]].keys()[0]
	elasticsearch_tokens.append(longest_substring)

	original_term, pre, post = extract_longest_substring(longest_substring)
	tokens = rebuild_tokens(original_term,longest_substring,pre,post)

	parse_into_tokens(longest_substrings, terms, pre + " " + post, True)

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
parse_into_tokens(longest_substrings, terms, input, recursive)
display_results()
