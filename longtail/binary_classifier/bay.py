#!/usr/local/bin/python3
# pylint: disable=all

from longtail.custom_exceptions import InvalidArguments

def predict_if_physical_transaction(description=None):
	""" Model for binary classification specific to Bay Area """

	import re

	tm_tokens = 'tokens_only'
	tm_full_term = 'full_terms_only'
	tm_all = 'all'

	def term_matches(text, field_name, term):
		""" Counts the number of occurences of term and its variants in text"""

		forms_list = term_forms[field_name].get(term, [term])
		options = term_analysis[field_name]
		token_mode = options.get('token_mode', tm_tokens)
		case_sensitive = options.get('case_sensitive', False)
		first_term = forms_list[0]
		if token_mode == tm_full_term:
			return full_term_match(text, first_term, case_sensitive)
		else:
			# In token_mode='all' we will match full terms using equals and
			# tokens using contains
			if token_mode == tm_all and len(forms_list) == 1:
				pattern = re.compile(r'^.+\b.+$', re.U)
				if re.match(pattern, first_term):
					return full_term_match(text, first_term, case_sensitive)
			return term_matches_tokens(text, forms_list, case_sensitive)

	def full_term_match(text, full_term, case_sensitive):
		""" Counts the match for full terms according to the case_sensitive option """
		if not case_sensitive:
			text = text.lower()
			full_term = full_term.lower()
		return 1 if text == full_term else 0

	def get_tokens_flags(case_sensitive):
		""" Returns flags for regular expression matching depending on text analysis options """
		flags = re.U
		if not case_sensitive:
			flags = (re.I | flags)
		return flags

	def term_matches_tokens(text, forms_list, case_sensitive):
		""" Counts the number of occurences of the words in forms_list in the text """
		flags = get_tokens_flags(case_sensitive)
		expression = r'(\b|_)%s(\b|_)' % '(\\b|_)|(\\b|_)'.join(forms_list)
		pattern = re.compile(expression, flags=flags)
		matches = re.findall(pattern, text)
		return len(matches)

	term_analysis = {
		"description": {
			"token_mode": 'all',
			"case_sensitive": False
		}
	}

	term_forms = {
		"description": {}
	}

	if (description is None):
		return '1'
	if (term_matches(description, "description", "purchase") > 0):
		return '1'
	if (term_matches(description, "description", "purchase") <= 0):
		if (term_matches(description, "description", "com") > 0):
			return '0'
		if (term_matches(description, "description", "com") <= 0):
			if (term_matches(description, "description", "800") > 0):
				return '0'
			if (term_matches(description, "description", "800") <= 0):
				if (term_matches(description, "description", "866") > 0):
					return '0'
				if (term_matches(description, "description", "866") <= 0):
					if (term_matches(description, "description", "san") > 0):
						return '1'
					if (term_matches(description, "description", "san") <= 0):
						if (term_matches(description, "description", "caus") > 0):
							return '1'
						if (term_matches(description, "description", "caus") <= 0):
							if (term_matches(description, "description", "ca") > 0):
								if (term_matches(description, "description", "fitness") > 0):
									return '0'
								if (term_matches(description, "description", "fitness") <= 0):
									if (term_matches(description, "description", "sunnyvale") > 0):
										return '1'
									if (term_matches(description, "description", "sunnyvale") <= 0):
										if (term_matches(description, "description", "camino") > 0):
											return '1'
										if (term_matches(description, "description", "camino") <= 0):
											return '1'
							if (term_matches(description, "description", "ca") <= 0):
								if (term_matches(description, "description", "checkcard") > 0):
									if (term_matches(description, "description", "inc") > 0):
										return '0'
									if (term_matches(description, "description", "inc") <= 0):
										return '0'
								if (term_matches(description, "description", "checkcard") <= 0):
									return '0'

def process_list(transactions=None):
	physical = []
	non_physical = []

	for row in transactions:
		prediction = predict_if_physical_transaction(row)
		if prediction == "1":
			physical.append(row)
		elif prediction == "0":
			non_physical.append(row)
		else:
			logging.info("Unable to classify: " + row)

	return physical

if __name__ == "__main__":

	import sys, logging

	if len(sys.argv) != 2:
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)
	try:
		input_file = open(sys.argv[1], encoding='utf-8')
		trans_list = input_file.read().splitlines()
		process_list(trans_list)	
		input_file.close()
	except FileNotFoundError:
		print(sys.argv[1], " not found, aborting.")
		logging.error(sys.argv[1] + " not found, aborting.")
		sys.exit()							
