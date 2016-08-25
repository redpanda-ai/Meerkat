import os
import logging
from pybloom import ScalableBloomFilter
from meerkat.various_tools import load_params

directories = [obj[0] for obj in os.walk('meerkat/classification/bloom_filter/assets/merchants')]

def create_location_bloom(src_dirs, dst_filename):
	"""Creates a bloom filter from the provided input file."""
	sbf = ScalableBloomFilter(initial_capacity=100000, error_rate=0.001,\
		mode=ScalableBloomFilter.SMALL_SET_GROWTH)

	for directory in src_dirs:
		if directory.split('/')[-1].startswith('merchant') or directory.split('/')[-1].startswith('Bealls'):
			continue
		cities = load_params(directory + '/unique_city.json')
		merchant_name = directory.split('/')[-1].upper()
		if merchant_name.startswith('COSTCO'): merchant_name = 'COSTCO'
		for city in cities:
			sbf.add(city + ' ' + merchant_name)

	with open(dst_filename, "bw+") as location_bloom:
		sbf.tofile(location_bloom)

	return sbf

def get_location_bloom():
	"""Attempts to fetch a bloom filter from a file, making a new bloom filter
	if that is not possible."""
	sbf = None
	bf_filename = "meerkat/classification/bloom_filter/assets/unique_city_bloom"
	try:
		sbf = ScalableBloomFilter.fromfile(open(bf_filename, "br"))
		logging.info("Location bloom filter loaded from file.")
	except:
		logging.info("Creating new bloom filter of unique city")
		sbf = create_location_bloom(directories, bf_filename)
	return sbf

def get_city_dict():
	unique_city_state_dict = dict()
	top_merchants = set()
	for directory in directories:
		if directory.split('/')[-1].startswith('merchant') or directory.split('/')[-1].startswith('Bealls'):
			continue
		merchant_name = directory.split('/')[-1].upper()
		if merchant_name.startswith('COSTCO'): merchant_name = 'COSTCO'
		top_merchants.add(merchant_name)
		unique_city_state_dict[merchant_name] = load_params(directory + '/unique_city_state.json')
	return unique_city_state_dict, top_merchants

city_bloom = get_location_bloom()
unique_city_state_dict, top_merchants = get_city_dict()

def location_from_merchant(text, merchant):
	merchant = ''.join(merchant.upper().split())
	if merchant not in top_merchants: return None
	transaction_word = ['PURCHASE']
	text = text.upper()
	for i in range(len(text) - 1, -1, -1):
		for j in range(i, -1, -1):
			city = text[j: i + 1]
			if city + ' ' + merchant in city_bloom and city in unique_city_state_dict[merchant] and city not in transaction_word:
				return (city, unique_city_state_dict[merchant][city])
	return None


if __name__ == "__main__":
	my_location_bloom = get_location_bloom()
	print(location_from_merchant('COSTCO WHOLESALE    ATLANTA 000009904   7704311702 ', 'COSTCO'))
