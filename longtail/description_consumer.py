'''
Created on Jan 14, 2014

@author: J. Andrew Key
@author: Matt Sevrens
'''

#!/bin/python3
# pylint: disable=R0914

import copy, elasticsearch, hashlib, json, logging, threading, queue, re
from scipy.stats.mstats import zscore

from longtail.custom_exceptions import UnsupportedQueryType
from longtail.query_templates import (GENERIC_ELASTICSEARCH_QUERY, STOP_WORDS,
	get_match_query, get_qs_query)
from longtail.various_tools import string_cleanse

class DescriptionConsumer(threading.Thread):
	''' Acts as a client to an ElasticSearch cluster, tokenizing description
	strings that it pulls from a synchronized queue. '''

	STILL_BREAKABLE = 2

	def __begin_parse(self):
		"""Creates data structures used the first call into the
		__parse_into_search_tokens function."""
		logger = logging.getLogger("thread " + str(self.thread_id))
		#Abort processing, if input string is None
		if self.input_string != None:
			logger.info("Input String " + self.input_string)
		else:
			logger.warning("No input string provided, skipping")
			return False
		self.recursive = False
		self.my_meta["terms"] = []
		self.my_meta["long_substrings"] = {}
		self.__parse_into_search_tokens(self.input_string, self.recursive)
		return True

	def __display_results(self):
		"""Displays our tokens, n-grams, and search results."""
		phone_re = re.compile("^[0-9/#]{10}$")
		numeric = re.compile("^[0-9/#]+$")

		stop_tokens = []
		filtered_tokens = []
		numeric_tokens = []
		addresses = []
		phone_numbers = []
		tokens = self.my_meta["tokens"]
		unigram_tokens = self.my_meta["unigram_tokens"]
		for token in tokens:
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
		logger = logging.getLogger("thread " + str(self.thread_id))
		logger.info("TOKENS ARE: " + str(tokens))
		logger.info("Unigrams are:\n\t" + str(tokens))
		logger.info("Unigrams matched to ElasticSearch:\n\t" + str(unigram_tokens))
		logger.info("Of these:")
		logger.info("\t" + str(len(stop_tokens)) + " stop words:      "\
		+ str(stop_tokens))
		logger.info("\t" + str(len(phone_numbers)) + " phone_numbers:   "\
		+ str(phone_numbers))
		logger.info("\t" + str(len(numeric_tokens)) + " numeric words:   "\
		+ str(numeric_tokens))
		logger.info("\t" + str(len(filtered_tokens)) + " unigrams: "\
		+ str(filtered_tokens))

		count, matching_address = self.__get_matching_address()
		if count > 0:
			addresses.append(matching_address)
		logger.info("\t" + str(len(addresses)) + " addresses: " + str(addresses))

		#show all search terms separated by spaces
		query_string = " ".join(filtered_tokens)
		self.__get_n_gram_tokens(filtered_tokens)

		matched_n_gram_tokens = self.__search_n_gram_tokens()
		logger.info("\t" + str(len(matched_n_gram_tokens)) + " 2+ grams: "\
		+ str(matched_n_gram_tokens))

		self.__generate_final_query(query_string, matching_address\
		, phone_numbers, matched_n_gram_tokens)

		#TODO: a tail recursive search for longest matching
		#n-gram on a per composite-feature basis.

	def __display_search_results(self, search_results):
		"""Displays search results."""
		hits = search_results['hits']['hits']
		scores, results, fields_found = [], [], []
		params = self.params
		for hit in hits:
			hit_fields, score = hit['fields'], hit['_score']
			scores.append(score)
			field_order = params["output"]["results"]["fields"]
			fields_in_hit = [field for field in hit_fields]
			ordered_hit_fields = []
			for ordinal in field_order:
				if ordinal in fields_in_hit:
					my_field = str(hit_fields[ordinal])
					fields_found.append(ordinal)
					ordered_hit_fields.append(my_field)
				else:
					fields_found.append(ordinal)
					ordered_hit_fields.append("")
			results.append(\
			"[" + str(round(score, 3)) + "] " + " ".join(ordered_hit_fields))

		self.__display_z_score_delta(scores)
		for result in results:
			print(result)

	def output_to_result_queue(self, search_results):
		"""Decides whether to output and pushes to result_queue"""
		hits = search_results['hits']['hits']
		scores, fields_found = [], []
		output_dict = {}
		params = self.params
		parameter_key = self.parameter_key
		field_order = params["output"]["results"]["fields"]
		top_hit = hits[0]
		hit_fields = top_hit["fields"]
		fields_in_hit = [field for field in hit_fields]
		ordered_hit_fields = []

		for hit in hits:
			scores.append(hit['_score'])

		for ordinal in field_order:
			if ordinal in fields_in_hit:
				my_field = str(hit_fields[ordinal])
				fields_found.append(ordinal)
				ordered_hit_fields.append(my_field)
			else:
				fields_found.append(ordinal)
				ordered_hit_fields.append("")

		z_score_delta = self.__display_z_score_delta(scores)
		top_score = top_hit['_score']

		#Unable to generate z_score
		if z_score_delta == None:
			output_dict = dict(zip(fields_found, ordered_hit_fields))
			output_dict['DESCRIPTION'] = self.input_string
			self.result_queue.put(output_dict)
			return

		#Send to result Queue if score good enough
		if z_score_delta > float(parameter_key.get("z_score_threshold", "2")):
			output_dict = dict(zip(fields_found, ordered_hit_fields))
		else:
			output_dict = dict(zip(fields_found, ([""] * len(fields_found))))

		output_dict['DESCRIPTION'] = self.input_string
		self.result_queue.put(output_dict)

		logging.info("Z_SCORE_DELTA: " + str(z_score_delta))
		logging.info("TOP_SCORE: " + str(top_score))

	def __display_z_score_delta(self, scores):
		"""Display the Z-score delta between the first and second scores."""
		logger = logging.getLogger("thread " + str(self.thread_id))
		if len(scores) < 2:
			logger.info("Unable to generate Z-Score")
			return None

		z_scores = zscore(scores)
		first_score, second_score = z_scores[0:2]
		z_score_delta = round(first_score - second_score, 3)
		logger.info("Z-Score delta: [" + str(z_score_delta) + "]")
		quality = "Non"
		if z_score_delta <= 1:
			quality = "Low-grade"
		elif z_score_delta <= 2:
			quality = "Mid-grade"
		else:
			quality = "High-grade"

		logger.info("Top Score Quality: " + quality)
		return z_score_delta

	def __extract_longest_substring(self, longest_substring):
		"""Extracts the longest substring from our input, returning left
		over fragments to find the following strings:
		1.  "original_term": the original string containing "longest_substring"
		2.  "pre": substring preceding "longest_substring"
		3.  "post" : substring following "longest_substring" """

		long_substrings = self.my_meta["long_substrings"]
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

	def __init__(self, thread_id, params, desc_queue, result_queue, parameter_key):
		''' Constructor '''
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.desc_queue = desc_queue
		self.result_queue = result_queue
		self.input_string = None
		self.params = params
		self.parameter_key = parameter_key
		cluster_nodes = self.params["elasticsearch"]["cluster_nodes"]
		self.es_connection = elasticsearch.Elasticsearch(cluster_nodes,
			sniff_on_start=True, sniff_on_connection_fail=True, sniffer_timeout=3)
		self.recursive = False
		self.my_meta = None
		self.n_gram_tokens = None
		self.__reset_my_meta()
		self.__set_logger()

	def __generate_final_query(self, qs_query, address, phone_numbers
	, matching_n_grams):
		"""Constructs a complete boolean query based upon:
		1.  Unigrams (query_string)
		2.  Addresses (match)
		3.  Phone Number (match)"""
		#Add dynamic output based upon a dictionary of token types
		#See comment above
		logger = logging.getLogger("thread " + str(self.thread_id))
		my_meta = self.my_meta
		search_components = []
		parameter_key = self.parameter_key
		field_boosts = ["_all^1"]
		field_boosts.append("BUSINESSSTANDARDNAME^"\
		+ parameter_key.get("business_name_boost", "1"))
		address_boost = "composite.address^" + parameter_key.get("address_boost", "1")
		phone_boost = "composite.phone^" + parameter_key.get("phone_boost", "1")

		logger.info("Search components are:")
		logger.info("\tUnigrams: '" + qs_query + "'")
		search_components.append((qs_query, "qs_query", field_boosts, 1))
		if address is not None:
			logger.info("\tMatching 'Address': '" + address + "'")
			search_components.append((address, "match_query"\
			, [address_boost], 10))
		if len(phone_numbers) != 0:
			for phone_num in phone_numbers:
				logger.info("\tMatching 'Phone': '" + phone_num + "'")
				search_components.append((phone_num, "match_query"\
				, [phone_boost], 1))
		for n_gram in matching_n_grams:
			search_components.append((n_gram, "match_query"\
			, ["_all^1"], 1))

		my_obj = self.__get_boolean_search_object(search_components)
		logger.info(json.dumps(my_obj))
		my_results = self.__search_index(my_obj)
		metrics = my_meta["metrics"]
		logger.warning("Cache Hit / Miss: " + str(metrics["cache_count"])\
		+ " / " + str(metrics["query_count"]))
		self.__display_search_results(my_results)
		logger.critical(str(my_results))
		self.output_to_result_queue(my_results)

	def __get_boolean_search_object(self, search_components):
		"""Builds an object for a "bool" search."""
		params = self.params
		parameter_key = self.parameter_key
		bool_search = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
		bool_search["fields"] = params["output"]["results"]["fields"]
		bool_search["from"] = 0
		bool_search["size"] = parameter_key.get("es_result_size", "10")

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
				+ " for a query of type: " + query_type)
			bool_search["query"]["bool"]["should"].append(my_subquery)
		return bool_search

	def __get_composite_search_count(self, list_of_ngrams, feature_name):
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
			total = self.__search_index(my_new_query)['hits']['total']
			if total == 0:
				my_new_query["query"]["bool"]["should"] = []
			else:
				return total, term
		return 0, None

	def __get_matching_address(self):
		"""Sadly, this function is temporary.  I plan to go to a more
		generic approach that exhaustively works with all n-grams
		against all composite features."""
		numeric = re.compile("^[0-9]+$")
		address_candidates = {}
		n_gram = []
		my_meta = self.my_meta
		tokens = my_meta["tokens"]
		for token in tokens:
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
			count, term = self.__get_composite_search_count(candidate_list\
			, "composite.address")
			if count > 0:
				return count, term
		return 0, None

	def __get_n_gram_tokens(self, list_of_tokens):
		"""Generates a list of n-grams where n >= 2."""
		unigram_size = 1
		self.n_gram_tokens = {}
		end_index = len(list_of_tokens)
		for end_offset in range(end_index):
			for start_index in range(end_index-end_offset):
				n_gram_size = end_index - end_offset - start_index
				if n_gram_size > unigram_size:
					if n_gram_size not in self.n_gram_tokens:
						self.n_gram_tokens[n_gram_size] = []
						new_n_gram = \
						list_of_tokens[start_index:end_index-end_offset]
						self.n_gram_tokens[n_gram_size].append(" ".join(new_n_gram))
		return

	def __parse_into_search_tokens(self, input_string, recursive):
		"""Recursively attempts to parse an unstructured transaction
		description into TOKENS.
		Example: "MEL'S DRIVE-IN #2 SAN FRANCISCOCA 24492153337286434101508" """
		#This loop breaks up the input string looking for new search terms
		#that match portions of our ElasticSearch index
		#my_meta = self.my_meta
		tokens = self.my_meta["tokens"]
		long_substrings = self.my_meta["long_substrings"]
		terms = self.my_meta["terms"]
		if len(input_string) >= DescriptionConsumer.STILL_BREAKABLE:
			new_terms = input_string.split()
			for term in new_terms:
				term = string_cleanse(term)
				if not recursive:
					tokens.append(term)
				substrings = {}
				self.__powerset(term, substrings)
				big_substring = self.__search_substrings(substrings)
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
		self.my_meta["unigram_tokens"].append(longest_substring)

		original_term, pre, post = self.__extract_longest_substring(longest_substring)
		self.__rebuild_tokens(original_term, longest_substring, pre, post)
		self.__parse_into_search_tokens(pre + " " + post, True)

	def __powerset(self, term, substrings):
		"""Recursively discover all substrings for a term."""
		term_length = len(term)
		if term_length not in substrings:
			substrings[term_length] = {}
		if term not in substrings[term_length]:
			substrings[term_length][term] = ""
		if term_length <= DescriptionConsumer.STILL_BREAKABLE:
			return
		if term in STOP_WORDS:
			return
		else:
			self.__powerset(term[0:-1], substrings)
			self.__powerset(term[1:], substrings)

	def __rebuild_tokens(self, original_term, longest_substring, pre, post):
		"""Rebuilds our complete list of tokens following a substring
		extraction."""
		tokens = self.my_meta["tokens"]
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
		self.my_meta["tokens"] = rebuilt_tokens

	def __reset_my_meta(self):
		"""Purges several object data structures and re-initializes them."""
		self.recursive = False
		self.n_gram_tokens = {}
		self.my_meta = {}
		self.my_meta["unigram_tokens"] = []
		self.my_meta["tokens"] = []
		self.my_meta["metrics"] = {"query_count" : 0, "cache_count" : 0}

	def __search_index(self, input_as_object):
		"""Searches the merchants index and the merchant mapping"""
		logger = logging.getLogger("thread " + str(self.thread_id))
		input_data = json.dumps(input_as_object).encode('UTF-8')
		#Check the cache first
		hash_object = hashlib.md5(str(input_data).encode())
		input_hash = hash_object.hexdigest()
		#input_hash = str(input_data)
		if input_hash in self.params["search_cache"]:
			logger.info("Cache hit, short-cutting")
			self.my_meta["metrics"]["cache_count"] += 1
			output_data = self.params["search_cache"][input_hash]
		else:
			logger.info("Cache miss, searching")
			try:
				output_data = self.es_connection.search(
					index=self.params["elasticsearch"]["index"], body=input_as_object)
			except Exception:
				logging.critical("Unable to process the following: " + str(input_as_object))
				output_data = '{"hits":{"total":0}}'

		metrics = self.my_meta["metrics"]
		metrics["query_count"] += 1
		self.params["search_cache"][input_hash] = output_data
		return output_data

	def __search_n_gram_tokens(self):
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

		logger = logging.getLogger("thread " + str(self.thread_id))
		logger.info("Matched the following n-grams to our merchants index:")
		for key in reversed(sorted(self.n_gram_tokens.keys())):
			for term in self.n_gram_tokens[key]:
				sub_query["match"]["_all"]["query"] = term
				my_new_query["query"]["bool"]["should"].append(sub_query)
				hit_count = self.__search_index(my_new_query)['hits']['total']
				del my_new_query["query"]["bool"]["should"][0]
				if hit_count > 0:
					logger.info(str(key) + "-gram : " + term + " (" + str(hit_count) + ")")
					matched_n_gram_tokens.append(term)
		return matched_n_gram_tokens

	def __search_substrings(self, substrings):
		"""Looks for the longest substring in our merchants index that
		can actually be found."""
		my_new_query = copy.deepcopy(GENERIC_ELASTICSEARCH_QUERY)
		my_new_query["size"], my_new_query["from"] = 0, 0

		sub_query = {}
		sub_query["query_string"] = {}
		sub_query["query_string"]["query"] = "__term"
		my_new_query["query"]["bool"]["should"].append(sub_query)

		for key in reversed(sorted(substrings.keys())):
			for term in substrings[key]:
				#These words are reserved and will cause HTTP Response errors if
				#sent as-is to an ElasticSearch query
				if term in ["AND", "OR"]:
					term = term.lower()
				hit_count = 0
				if len(term) > 1:
					sub_query["query_string"]["query"] = str(term)
					hit_count = self.__search_index(my_new_query)['hits']['total']
				if hit_count > 0:
					return term

	def __set_logger(self):
		"""Creates a logger, based upon the supplied config object."""

		levels = {'debug': logging.DEBUG, 'info': logging.INFO\
		, 'warning': logging.WARNING, 'error': logging.ERROR\
		, 'critical': logging.CRITICAL}
		params = self.params
		my_level = params["logging"]["level"]
		if my_level in levels:
			my_level = levels[my_level]
		my_path = params["logging"]["path"]
		my_formatter = logging.Formatter(params['logging']['formatter'])
		#You'll want to add something to identify the thread
		my_logger = logging.getLogger("thread " + str(self.thread_id))
		my_logger.setLevel(my_level)
		file_handler = logging.FileHandler(my_path)
		file_handler.setLevel(my_level)
		file_handler.setFormatter(my_formatter)
		my_logger.addHandler(file_handler)

		#Add console logging, if configured
		my_console = params["logging"]["console"]
		if my_console is True:
			console_handler = logging.StreamHandler()
			console_handler.setLevel(my_level)
			console_handler.setFormatter(my_formatter)
			my_logger.addHandler(console_handler)

		my_logger.info("Log initialized.")
		params_json = json.dumps(params, sort_keys=True, indent=4\
		, separators=(',', ': '))
		my_logger.info(params_json)

	def run(self):
		while True:
			try:
				self.input_string = self.desc_queue.get()
				self.__begin_parse()
				self.__display_results()
				self.__reset_my_meta()
				self.desc_queue.task_done()

			except queue.Empty:
				print(str(self.thread_id), " found empty queue, terminating.")
		return True
