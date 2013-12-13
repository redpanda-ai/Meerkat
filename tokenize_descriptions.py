#!/usr/bin/python

import json, os, sys, re, urllib, urllib2
from subprocess import Popen, PIPE
from query_templates import unpack_attributes, unpack_json_id
from query_templates import BOOL_QUERY, STANDARD_QUERY

class InvalidArguments(Exception):
	pass

STILL_BREAKABLE = 2
UNIGRAM_SIZE = 1
PLACEHOLDER = "foo"
PLACEHOLDER_LENGTH = len(PLACEHOLDER) + len(" :")

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
	show_terms = " ".join(filtered_tokens)
	n_gram_tokens = get_n_gram_tokens(filtered_tokens)
	matched_n_gram_tokens = search_n_gram_tokens(n_gram_tokens)

	#FIXME: search the matched_n_gram_tokens with a "match" query using "bool" to combine them with the "term query"
	#It's currently not returning very relevant results, so it is commented out
	#build_boolean_search(matched_n_gram_tokens,show_terms)
	
	print "Elasticsearching the following"
	print "You can also try Google:\n\t" + str(show_terms)
	get_elasticsearch_results(show_terms)

def build_boolean_search(matched_n_gram_tokens,query_string):
	query_parts = []
	print "Building boolean search"
	print "Should match terms"
	query_key, query_values = "match._all", [("query",'"__term"'), ("type",'"phrase"'),("boost",0.01)]
	x = unpack_json_id(query_key, query_values)
	print "x is " 
	for term in matched_n_gram_tokens:
		y = x.replace("__term",term)
		query_parts.append(y)
	query_key, query_values = "query_string", [ ("query",'"__term"') ]
	x = unpack_json_id(query_key, query_values)
	y = x.replace("__term",term) 
	query_parts.append(y)
	term = ",".join(query_parts)

	should = '"should" : [ ' + term + ']'
	attributes = [ ("from",0), ("size",10)]
	query_values = [ ]
	input = build_bool_input(should,BOOL_QUERY,attributes,query_key,query_values)
	input_json = json.loads(input)
	#print input_json
	output = search_index_with_input(input)
	hits = output['hits']['hits']
	print "BEGIN Composite"
	display_hits(hits)
	print "END Composite"

def build_bool_input(should,action,attributes,query_key,query_values):
	input = action.replace("__attributes",(unpack_attributes(attributes)))
	input = input.replace("__should",should)
	return input

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
	query_key = "match._all"
	query_values = [ ("query",'"__term"'), ("type",'"phrase"') ]

	print "Matched the following n-grams to our merchants index:"
	for key in reversed(sorted(n_gram_tokens.iterkeys())):
		for term in n_gram_tokens[key]:
			attributes = [ ("size",0) ]
			hit_count = search_index(term,STANDARD_QUERY,attributes,query_key,query_values)['hits']['total']
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
def search_index(term,action,attributes,query_key,query_values):
	input = build_search_input(term,action,attributes,query_key,query_values)
	return search_index_with_input(input)

def build_search_input(term,action,attributes,query_key,query_values):
	input = action.replace("__attributes",(unpack_attributes(attributes)))
	input = input.replace("__query", unpack_json_id(query_key, query_values))
	input = input.replace("__term",urllib.quote_plus(term))
	return input


def search_index_with_input(input):
	url = "http://brainstorm8:9200/merchants/merchant/_search"
	request = urllib2.Request(url, input)
	response = urllib2.urlopen(request)
	output = json.loads(response.read())
	return output

#Runs an ElasticSearch query to find the results
def get_elasticsearch_results(unigram_tokens):
	query_key = "query_string"
	query_values = [ ("query",'"__term"') ]
	attributes = [ ("from",0), ("size",10) ]
	output = search_index(unigram_tokens,STANDARD_QUERY,attributes,query_key,query_values)
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
	query_key = "query_string"
	query_values = [ ("query",'"__term"') ]
	attributes = [ ("size",0) ]
	for key in reversed(sorted(substrings.iterkeys())):
		for term in substrings[key]:
			hit_count = 0
			if len(term) > 1:
				hit_count = search_index(term,STANDARD_QUERY,attributes,query_key,query_values)['hits']['total']
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

