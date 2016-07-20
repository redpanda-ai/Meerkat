import logging
from pybloom import ScalableBloomFilter
from meerkat.various_tools import load_params

MIN_LEN, MAX_LEN = None, None

def create_location_bloom(src_filename, dst_filename):
	"""Creates a bloom filter from the provided input file."""
	sbf = ScalableBloomFilter(initial_capacity=100000, error_rate=0.001,\
		mode=ScalableBloomFilter.SMALL_SET_GROWTH)

	cities = load_params(src_filename)

	for city in cities:
		sbf.add(city)
	with open(dst_filename, "bw+") as location_bloom:
		sbf.tofile(location_bloom)

	return sbf

def get_location_bloom():
	"""Attempts to fetch a bloom filter from a file, making a new bloom filter
	if that is not possible."""
	sbf = None
	bf_filename = "meerkat/classification/bloom_filter/assets/unique_city_bloom"
	json_filename = "meerkat/classification/bloom_filter/assets/Starbucks_unique_city.json"
	try:
		sbf = ScalableBloomFilter.fromfile(open(bf_filename, "br"))
		logging.info("Location bloom filter loaded from file.")
		test_bloom_filter(sbf)
	except:
		logging.info("Creating new bloom filter of unique city")
		sbf = create_location_bloom(json_filename, bf_filename)
	return sbf

def city_length_range():
	global MIN_LEN, MAX_LEN
	if MIN_LEN is None and MAX_LEN is None:
		json_filename = "meerkat/classification/bloom_filter/assets/Starbucks_unique_city.json"
		cities = load_params(json_filename)
		MIN_LEN, MAX_LEN = 100, 0
		for city in cities:
			MIN_LEN = min(MIN_LEN, len(city))
			MAX_LEN = max(MAX_LEN, len(city))
	return MIN_LEN, MAX_LEN

def test_bloom_filter(sbf):
	canadian_locations = [
		"TORONTO",
		"MONTREAL",
		"CALGARY",
		"OTTOWA",
		"EDMONTON"
	]
	us_locations = [
		"SAN FRANCISCO",
		"CHICAGO",
		"BOISE",
		"HIGHGATE FALLS",
		"SAN JOSE",
		"WEST MONROE",
		"DILLARD",
		"FAKE CITY",
		"CARSON CITY",
		"SAINT LOUIS",
		"SUNNYVALE"
	]
	logging.info("Touring Canada")
	line = "{0} in bloom filter: {1}"
	for location in canadian_locations:
		logging.info(line.format(location, location in sbf))
	logging.info("Touring United States")
	for location in us_locations:
		logging.info(line.format(location, location in sbf))

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	my_location_bloom = get_location_bloom()
	MIN_LEN, MAX_LEN = city_length_range()
