"""use trie to find location

Created on Jun 26, 2016
@author: Feifei Zhu
"""

import csv

from .generate_json import generate_js
from meerkat.various_tools import load_params

STATES = [
	"AL", "AK", "AZ", "AR", "CA",
	"CO", "CT", "DE", "FL", "GA",
	"HI", "ID", "IL", "IN", "IA",
	"KS", "KY", "LA", "ME", "MD",
	"MA", "MI", "MN", "MS", "MO",
	"MT", "NE", "NV", "NH", "NJ",
	"NM", "NY", "NC", "ND", "OH",
	"OK", "OR", "PA", "RI", "SC",
	"SD", "TN", "TX", "UT", "VT",
	"VA", "WA", "WV", "WI", "WY",
	"DC"]

SHORTENINGS = {
	"EAST": "E",
	"WEST": "W",
	"NORTH": "N",
	"SOUTH": "S",
	"SAINT": "ST",
	"FORT": "FT",
	"BEACH": "BCH",
}

LENGTHENINGS = {}
for key in SHORTENINGS.keys():
	LENGTHENINGS[SHORTENINGS[key]] = key

#We are doing something important: FIXME
SHORTENINGS["CITY"] = ""

class TrieNode():
	"""This is the most basic component data structure within a Trie"""
	def __init__(self):
		"""Initializes the TrieNode"""
		self.children = dict()
		self.isword = False

class Trie():
	"""Prefix-tree class used to search strings, similar in purpose to a dictionary but smaller."""
	def __init__(self):
		"""Initializes the Trie class."""
		self.root = TrieNode()

	def add(self, word):
		"""Adds a string to the Trie."""
		node = self.root
		state = word[:2]
		child = node.children.get(state)
		if not child:
			child = TrieNode()
			node.children[state] = child
		node = child
		word = word[2:]
		for letter in word:
			child = node.children.get(letter)
			if not child:
				child = TrieNode()
				node.children[letter] = child
			node = child
		node.isword = True

	def search(self, word):
		"""Searches to see if the 'word' is already in the Trie."""
		def find(node, word, path):
			"""The period character (.) signifies any letter, similar to the regex notation for .
			This tail-recursive function updates the word, the path, and the node at each level of
			recursion to ultimately find out whether the word is in the entire Trie."""
			any_char = "."

			if not word:
				if node.isword:
					result.append(path)
					return True
				else:
					return False
			if word[0] == any_char and node.isword:
				result.append(path)
				return True
			if word[0] == any_char:
				for child_key in node.children:
					if find(node.children[child_key], word[1:], path + child_key):
						return True
				return False
			else:
				child = node.children.get(word[0])
				if not child:
					return False
				return find(child, word[1:], path + word[0])
		result = []
		state = word[:2]
		if find(self.root.children[state], word[2:], state):
			return result[0]
		else:
			return False

def standardize(text):
	"""converts text to all caps, no punctuation, and no whitespace"""
	text = text.upper()
	text = ''.join(text.split())
	for mark in r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~':
		text = text.replace(mark, "")
	return text

def get_short_forms(city):
	"""Generate a list of all possible strings where we abbreviate certain words.
	For example: if city="WEST FORT WORTH",
		then return [ "WESTFORTWORTH", WESTFTWORTH, WFORTWORTH, WFTWORTH ] """
	city = city.upper().replace('.', '')
	tokens = city.split()
	token_length = len(tokens)
	token_lists = []
	def depth_first_search(token_index=0, token_list=[]):
		"""traverse the tokens and replace long terms with shortened or abbreviated terms"""
		#Base case, we are done
		if token_index == token_length:
			token_lists.append(token_list)
			return
		#move forward in our list of tokens
		token_index += 1
		#Generate the list where we SKIP abbreviations
		depth_first_search(token_index=token_index, token_list=token_list + [tokens[token_index]])
		#Generate the list where we ABBREVIATE, if we can
		if tokens[token_index] in SHORTENINGS:
			depth_first_search(token_index=token_index,
				token_list=token_list + [SHORTENINGS[tokens[token_index]]])

	#Find all possible short forms of the city name
	depth_first_search()
	#Return a list of strings
	return [''.join(token_list) for token_list in token_lists]

def get_longest_form(city):
	"""Takes any city string and returns the longest possible string, by lengthening all
	abbreviations."""
	city = city.upper()
	city = city.replace('.', '')
	tokens = city.split()
	for i, _ in enumerate(tokens):
		#Lengthen
		if tokens[i] in LENGTHENINGS:
			tokens[i] = LENGTHENINGS[tokens[i]]
		#Use correct capitalization
		if len(tokens[i]) > 1:
			tokens[i] = tokens[i][0] + tokens[i][1:].lower()

	return ' '.join(tokens)

def build_trie(csv_filename, json_filename):
	"""Builds a Trie using a CSV and a JSON file."""
	trie = Trie()
	city_map = dict()

	# add a CSV file (e.g. us_cities_larger.csv)
	csv_file = csv.reader(open(csv_filename, encoding="utf-8"), delimiter="\t")
	for row in csv_file:
		try:
			int(row[2]) # some of the rows don't acually record a state name
		except ValueError:
			city = row[2]
			state = row[4].upper()
			if state in STATES:
				short_forms = get_short_forms(city)
				for short_form in short_forms:
					trie.add(state + short_form)
					city_map[state + short_form] = (city, state)

	# add locations.json file
	locations = load_params(json_filename)
	states = locations["aggregations"]["states"]["buckets"]
	for state in states:
		state_name = state["key"].upper()
		for city in state["localities"]["buckets"]:
			city_name = get_longest_form(city["key"])
			short_forms = get_short_forms(city_name)
			for short_form in short_forms:
				trie.add(state_name + short_form)
				city_map[state_name + short_form] = (city_name, state_name)

	return trie, city_map

TRIE, MAP = build_trie("meerkat/classification/bloom_filter/assets/us_cities_larger.csv",
	'meerkat/classification/bloom_filter/assets/locations.json')

def get_biggest_match(my_string, use_wildcards=False):
	"""Return the largest match for my_string that is within the trie."""
	state_size, min_city_size = 2, 3
	min_string_length = state_size + min_city_size
	min_wildcard_length = min_string_length + 5

	if len(my_string) < min_string_length:
		return None

	state = my_string[-state_size:] #last two chars are the state
	biggest = None

	if not use_wildcards:
		for i in range(len(my_string) - min_string_length, -1, -1):
			city = my_string[i:-state_size]
			place = TRIE.search(state + city)
			if place:
				biggest = place
	else:
		for i in range(len(my_string) - min_wildcard_length, -1, -1):
			city = my_string[i:-state_size]
			place = TRIE.search(state + city + '....')
			if place:
				biggest = place

	return biggest

def location_split(text):
	tag = tag_text(text)
	text = standardize(text)
	json_file = 'meerkat/classification/bloom_filter/assets/words_start_with_states.json'
	try:
		open(json_file)
	except:
		generate_js()
	words = load_params(json_file)
	length = len(text)

	for i in range(length - 2, -1, -1):
		if (text[i:i+2] in STATES and tag[i+1] == 'C' and
			get_word(tag, text, i) not in words[text[i:i+2]] and not check_co_id(tag, text, i)):
			place = get_biggest_match(text[:i+2])
			if place:
				if place[2] in 'EWSN' and tag[i - (len(place) - 2)] == 'C': 
					plc = place[:2] + place[3:]
					if plc == TRIE.search(plc):
						place = plc
				try: return MAP[place]
				except: pass

	for i in range(length - 2, -1, -1):
		if text[i:i+2] in STATES and (i == length - 2 or tag[i + 2] == 'B'):
			place = get_biggest_match(text[:i+2], True)
			if place:
				try: return MAP[place]
				except: pass

	return None

def tag_text(text):
	'''make tag for text'''
	text = text.replace('\'', '')
	for mark in r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~':
		text = text.replace(mark, ' ')
	text = text.strip()
	tag = ''
	for i in range(len(text)):
		if text[i] == ' ':
			continue
		elif i == 0 or text[i - 1] == ' ':
			tag += 'B'
		else:
			tag += 'C'
	return tag

def get_word(tag, text, index):
	'''get the word concact with state'''
	end = index + 2
	for ch in tag[index + 2:]:
		if ch == 'C':
			end += 1
		else:
			break
	return text[index:end]

def check_co_id(tag, text, i):
	if text[i:i+2] == 'CO' and text[i+2:i+4] == 'ID' \
		and tag[i] == 'B' and tag[i+2] == 'B' and (i + 4 >= len(tag) or tag[i+4] == 'B'):
		return True
	elif text[i:i+2] == 'ID' and text[i-2:i] == 'CO' \
		and tag[i-2] == 'B' and (i+2 >= len(tag) or tag[i+2] == 'B'):
		return True
	else: return False

if __name__ == "__main__":
	print("This module is not meant to be run from the console.")

