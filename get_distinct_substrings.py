#!/usr/bin/python

import json, os, sys, re, urllib, urllib2
from subprocess import Popen, PIPE

def usage():
	print "Usage:\n\t<quoted_transaction_description_string>"

class InvalidArguments(Exception):
	pass

#class InvalidNumberOfLines(Exception):
#	pass

#class FileProblem(Exception):
#	pass

bad_characters = [ "\[", "\]", "'", "\{", "\}", '"', "/"]
x = "|".join(bad_characters)
cleanse_pattern = re.compile(x)

count_query = """
{
	"size" : 0,
	"query" : {
		"query_string" : {
			"query" : "__term"
		}
	}
}"""

show_query = """
{
	"from" : 0,
	"size" : 5,
	"fields" : [ 
		"BUSINESSSTANDARDNAME",
		"HOUSE",
		"STREET",
		"STRTYPE",
		"CITYNAME",
		"STATE",
		"ZIP",
		"pin.location" ],
	"query" : {
		"query_string" : {
			"query" : "__term"
		}
	}
}"""

search_terms = []
stop_words = ["CHECK", "CARD", "CHECKCARD", "PAYPOINT", "PURCHASE", "LLC" ]
expected_fields = [ "BUSINESSSTANDARDNAME", "HOUSE", "STREET", "STRTYPE", "CITYNAME", "STATE", "ZIP" ]
complex_expected_fields = [ "lat", "lon"]
def finish():
	numeric = re.compile("^[0-9/#]+$")

	junk_terms = []	
	filtered_terms = []
	numeric_terms = []
	for term in search_terms:
		#print term
		if term in stop_words:
			junk_terms.append(term)
		elif not numeric.search(term):
			filtered_terms.append(string_cleanse(term))
		else:
			numeric_terms.append(term)

	print "Search terms:" + str(search_terms)
	print "Junk terms:" + str(junk_terms)
	print "Numeric terms:" + str(numeric_terms)
	print "Filtered terms:" + str(filtered_terms)
		
	show_terms = " ".join(filtered_terms)	
	print "Sending the following terms to elasticsearch:\n\t" + str(show_terms)
	output = show_merchants(show_terms)
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

def start(longest_substrings, terms, input):
	if len(input) >= 2:
		new_terms = input.split()
		for term in new_terms:
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

	if len(longest_substrings) == 0:
		return	
	longest_substring = longest_substrings[sorted(longest_substrings.iterkeys())[-1]].keys()[0]
	l = len(longest_substring)

	search_terms.append(longest_substring)

	original_term = longest_substrings[len(longest_substring)][longest_substring]

	del longest_substrings[l][longest_substring]
	if len(longest_substrings[l]) == 0:
		del longest_substrings[l]
	
	new_input = original_term.replace(longest_substring,"",1)	
	start(longest_substrings, terms, new_input)

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

def count_merchants(term):
	url = "http://brainstorm8:9200/merchants/merchant/_search"
	input = count_query.replace("__term",urllib.quote_plus(term))
	request = urllib2.Request(url, input)
	response = urllib2.urlopen(request)
	output = json.loads(response.read())
	return output['hits']['total']

def show_merchants(term):
	url = "http://brainstorm8:9200/merchants/merchant/_search"
	input = show_query.replace("__term",urllib.quote_plus(term))
	request = urllib2.Request(url, input)
	response = urllib2.urlopen(request)
	output = json.loads(response.read())
	return output

def search_substrings(substrings):
	for key in reversed(sorted(substrings.iterkeys())):
		for term in substrings[key]:
			hit_count = 0
			if len(term) > 1:
				hit_count = count_merchants(term)
			if hit_count > 0:
				print "Found: " + term + " " + str(hit_count) + " times."
				#print hit_count
				return term

def initialize():
	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments("Incorrect number of arguments")
	return sys.argv[1]

def string_cleanse(s):
	return re.sub(cleanse_pattern,"",s)	

	
#input = "MEL'S DRIVE-IN #2 SAN FRANCISCOCA 24492153337286434101508"

input = initialize()

longest_substrings, terms = {}, []
print "Input: " + input
start(longest_substrings, terms, input)
finish()
