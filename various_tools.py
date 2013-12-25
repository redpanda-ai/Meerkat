#!/usr/bin/python

"""This is where we are keeping functions that are useful enough to call from
within multiple scripts."""

import collections, re
import numpy as np

def string_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [ "\[", "\]", "'", "\{", "\}", '"', "/" ]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)

def numeric_cleanse(original_string):
	"""Strips out characters that might confuse ElasticSearch."""
	bad_characters = [ "\[", "\]", "'", "\{", "\}", '"', "/", "-" ]
	bad_character_regex = "|".join(bad_characters)
	cleanse_pattern = re.compile(bad_character_regex)
	return re.sub(cleanse_pattern, "", original_string)

def convert_to_non_unicode(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert_to_non_unicode, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert_to_non_unicode, data))
    else:
        return data

def z_score(a):
    """ Returns a 1D array of z-scores, one for each score in the passed array,
	computed relative to the passed array.  """
    mu = np.mean(a)
    sigma = np.std(a)
    return (np.array(a)-mu)/sigma


