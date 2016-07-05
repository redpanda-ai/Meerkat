"""use trie to find location

Created on Jun 26, 2016
@author: Feifei Zhu
"""

import json
import csv
import pandas as pd

from .generate_json import generate_js

STATES = [
	"AL", "AK", "AZ", "AR", "CA", \
	"CO", "CT", "DE", "FL", "GA", \
	"HI", "ID", "IL", "IN", "IA", \
	"KS", "KY", "LA", "ME", "MD", \
	"MA", "MI", "MN", "MS", "MO", \
	"MT", "NE", "NV", "NH", "NJ", \
	"NM", "NY", "NC", "ND", "OH", \
	"OK", "OR", "PA", "RI", "SC", \
	"SD", "TN", "TX", "UT", "VT", \
	"VA", "WA", "WV", "WI", "WY", \
	"DC" ]

SUBS = {
	# Directions
	"EAST": "E",
	"WEST": "W",
	"NORTH": "N",
	"SOUTH": "S",

	# Abbreviations
	"CITY" : "",
	"SAINT": "ST",
	"FORT": "FT",
	"BEACH": "BCH"

	# can get more in the future
}

FORMAT = {
	"E": "EAST",
	"W": "WEST",
	"N": "NORTH",
	"S": "SOUTH",

	"ST": "SAINT",
	"FT": "FORT",
	"BCH": "BEACH"
}

class TrieNode():
	def __init__(self):
		self.children = dict()
		self.isword = False

class Trie():
	def __init__(self):
		self.root = TrieNode()

	def add(self, word):
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
		def find(node, word, path):
			if not word: 
				if node.isword:
					res.append(path)
					return True
				else:
					return False
			if word[0] == '.' and node.isword:
				res.append(path)
				return True
			if word[0] == '.':
				for x in node.children:
					if find(node.children[x], word[1:], path + x): return True
				return False
			else:
				child = node.children.get(word[0])
				if not child: return False
				return find(child, word[1:], path + word[0])
		res = []
		state = word[:2]
		if find(self.root.children[state], word[2:], state):
			return res[0]
		else: return False

def get_json_from_file(input_filename):
	"""Opens a file of JSON and returns a json object"""
	try:
		input_file = open(input_filename, encoding='utf-8')
		my_json = json.loads(input_file.read())
		input_file.close()
		return my_json
	except IOError:
		print("{0} not found, aborting.".format(input_filename))
		sys.exit()
	return None

def standardize(text):
	"""converts text to all caps, no punctuation, and no whitespace"""
	text = text.upper()
	text = ''.join(text.split())
	for mark in '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~':
		text = text.replace(mark, "")
	return text

def get_city_subs(city):
	city = city.upper()
	city = city.replace('.', '')
	parts = city.split()
	length = len(parts)
	def dfs(step, path):
		if step == length:
			subs.append(path)
			return
		if parts[step] not in SUBS:
			dfs(step + 1, path + [parts[step]])
		else:
			dfs(step + 1, path + [parts[step]])
			dfs(step + 1, path + [SUBS[parts[step]]])
	subs = []
	dfs(0, [])
	subs = [''.join(sub) for sub in subs]
	return subs

def format_city_name(city):
	city = city.upper()
	city = city.replace('.', '')
	parts = city.split()
	for i in range(len(parts)):
		if parts[i] in FORMAT: parts[i] = FORMAT[parts[i]]
		parts[i] = parts[i][0] + parts[i][1:].lower() if len(parts[i]) > 1 else parts[i]
	return ' '.join(parts)

def build_trie(csv_filename, json_filename):
	trie = Trie()
	city_map = dict()

	# add us_cities_larger.csv file
	csv_file = csv.reader(open(csv_filename, encoding="utf-8"), delimiter="\t")
	for row in csv_file:
		try:
			int(row[2]) # some of the rows don't acually record a state name
		except ValueError:
			city = row[2]
			state = row[4].upper()
			if state in STATES:
				subs = get_city_subs(city)
				for sub in subs:
					trie.add(state + sub)
					city_map[state + sub] = (city, state)

	# add locations.json file
	locations = get_json_from_file(json_filename)
	states = locations["aggregations"]["states"]["buckets"]
	for state in states:
		state_name = state["key"].upper()
		for city in state["localities"]["buckets"]:
			city_name = format_city_name(city["key"])
			subs = get_city_subs(city_name)
			for sub in subs:
				trie.add(state_name + sub)
				city_map[state_name + sub] = (city_name, state_name)

	return trie, city_map

TRIE, MAP = build_trie("meerkat/classification/bloom_filter/assets/us_cities_larger.csv", 'meerkat/classification/bloom_filter/assets/locations.json')

def in_trie(text, enrich=False):
	if len(text) < 5:
		return None
	else:
		state = text[-2:]
		biggest = None

		if not enrich:
			for i in range(len(text) - 5, -1, -1):
				city = text[i:-2]
				place = TRIE.search(state + city)
				if place: biggest = place

		if not biggest and enrich:
			for i in range(len(text) - 10, -1, -1):
				city = text[i:-2]
				place = TRIE.search(state + city + '....')
				if place: biggest = place

	return biggest

def location_split(text):
	tag = tag_text(text)
	text = standardize(text)
	try:
		open('meerkat/classification/bloom_filter/assets/words_start_with_states.json')
	except:
		generate_js()
	words = get_json_from_file('meerkat/classification/bloom_filter/assets/words_start_with_states.json')
	length = len(text)

	for i in range(length - 2, -1, -1):
		if text[i:i+2] in STATES and tag[i+1] == 'C' and get_word(tag, text, i) not in words[text[i:i+2]]:
			place = in_trie(text[:i+2])
			if place:
				if place[2] in 'EWSN' and tag[i - (len(place) - 2)] == 'C': 
					plc = place[:2] + place[3:]
					if plc == TRIE.search(plc):
						place = plc
				try: return MAP[place]
				except: pass

	for i in range(length - 2, -1, -1):
		if text[i:i+2] in STATES and (i == length - 2 or tag[i + 2] == 'B'):
			place = in_trie(text[:i+2], True)
			if place:
				try: return MAP[place]
				except: pass

	return None

def tag_text(text):
	'''make tag for text'''
	text = text.replace('\'', '')
	for mark in '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~':
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

'''
def main():
	"""runs the file"""
	print("find_entities")
	input_file = "meerkat/classification/bloom_filter/input_file.txt.gz"
	# input_file = "data/input/should_search_labels.csv.gz"
#	input_file = sys.argv[1]
	data_frame = pd.io.parsers.read_csv(input_file, sep="|", compression="gzip")
	descriptions = data_frame['DESCRIPTION_UNMASKED']
	location_bloom_results = descriptions.apply(location_split)
	#TODO: add merchant results
	#merchant_bloom_results = descriptions.apply(merchant_split, axis=0)

	combined = pd.concat([location_bloom_results, descriptions], axis=1)
	combined.columns = ['LOCATION', 'DESCRIPTION_UNMASKED']

	pd.set_option('max_rows', 10000)
	combined.to_csv("meerkat/classification/bloom_filter/entities1.csv", \
		mode="w", sep="|", encoding="utf-8")
	print(combined)
	print(location_bloom_results.describe())

def onetest():
	import collections
	counter = collections.defaultdict(int)
	with open('20160101_MPANEL_BANK1.6.3.txt') as f:
		for line in f:
			des = line.split('|')[-1].rstrip('\n')
			place = location_split(des)
			if place: counter[place] += 1
			print('{0} <====== {1}'.format(place, des))

	print('')
	print('total transactions: 1000000')
	print('find location transactions: {0}'.format(sum(counter.values())))
	print('distinct cities: {0}'.format(len(counter)))
	sorted_counter = sorted(counter.items(), key=lambda item: item[1], reverse=True)
	print('city ranking')
	for x in sorted_counter:
		print(x)

def twotest():
	count = 0
	csv_file = csv.reader(open('GeoGroundTruthValidated.csv', encoding="utf-8"))
	for row in csv_file:
		des = row[11]
		city = row[13]
		state = row[14]
		expected = (city, state)
		result = location_split(des)
		if result != expected:
			count += 1
			print('{0} | {1} <====== {2}'.format(result, expected, des))

	print('accuracy: {0}'.format( (2652 - count) / 2652))
'''

if __name__ == "__main__":
#	main()
	print(location_split('altamonte spg fl'))

#	count = 0
#	csv_file = csv.reader(open('NoGeo.csv', encoding="utf-8"))
#	for row in csv_file:
#		des = row[11]
#		result = location_split(des)
#		if result: count += 1
#		print('{0} <====== {1}'.format(result, des))

#	print(count)

