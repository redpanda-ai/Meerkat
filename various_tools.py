#!/usr/bin/python

"""This is where we are keeping functions that are useful enough to call from
within multiple scripts."""

import re

print "hello"

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [ "\[", "\]", "'", "\{", "\}", '"', "/" ]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)	

