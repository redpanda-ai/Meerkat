#!/usr/bin/python

"""This is where we keep functions that are useful enough to call from
within multiple scripts."""

import re, csv

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	original_string = original_string.replace("OR", "or")
	original_string = original_string.replace("AND", "and")
	bad_characters = [r"\[", r"\]", r"\{", r"\}", r'"', r"/", r"\\", r"\:", r"\(", r"\)", r"-", r">", r"!", r"\*"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	with_spaces = re.sub(cleanse_pattern, " ", original_string)
	return ' '.join(with_spaces.split())

def numeric_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [r"\[", r"\]", r"'", r"\{", r"\}", r'"', r"/", r"-"]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)

def load_dict_list(file_name, encoding='utf-8', delimiter="|"):
	input_file = open(file_name, encoding=encoding, errors='replace')
	dict_list = list(csv.DictReader(input_file, delimiter=delimiter, quoting=csv.QUOTE_NONE))
	input_file.close()
	return dict_list
