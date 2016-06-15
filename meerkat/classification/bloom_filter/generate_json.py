import json

dictionary = {
	'AL': [
	],
	'AK': [
	],
	'AZ': [
	],
	'AR': [
	],
	'CA': [
		'CARD',
		'CAR'
	],
	'CO': [
		'COMPANY'
	],
	'CT': [
		'CTDI'
	],
	'DE': [
		'DEBIT'
	],
	'FL': [
	],
	'GA': [
	],
	'HI': [
	],
	'ID': [
	],
	'IL': [
	],
	'IN': [
		'INN'
	],
	'IA': [
	],
	'KS': [
	],
	'KY': [
	],
	'LA': [
	],
	'ME': [
	],
	'MD': [
		'MDEBIT'
	],
	'MA': [
	],
	'MI': [
	],
	'MN': [
	],
	'MS': [
	],
	'MO': [
	],
	'MT': [
	],
	'NE': [
	],
	'NV': [
	],
	'NH': [
	],
	'NJ': [
	],
	'NM': [
	],
	'NY': [
	],
	'NC': [
	],
	'ND': [
	],
	'OH': [
	],
	'OK': [
	],
	'OR': [
	],
	'PA': [
		'PAR',
		'PARDEBIT',
		'PARK',
		'PACIFIC'
	],
	'RI': [
		'RIVER'
	],
	'SC': [
	],
	'SD': [
	],
	'TN': [
	],
	'TX': [
	],
	'UT': [
	],
	'VT': [
	],
	'VA': [
	],
	'WA': [
	],
	'WV': [
	],
	'WI': [
		'WILD'
	],
	'WY': [
	],
}

def generate_js():
	with open('meerkat/classification/bloom_filter/assets/words_start_with_states.json', 'w') as fp:
		json.dump(dictionary, fp)

if __name__ == '__main__':
	generate_js()
