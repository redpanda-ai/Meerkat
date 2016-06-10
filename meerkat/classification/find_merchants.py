"""Finds merchants in given transaction strings

Created on June 30, 2015
@author: Sivan Mehta

USAGE
note: must be run from root of Meerkat

python3 -m meerkat.classification.find_merchants

output should be a table in the following form:
Detected        | Expected        | Input String
VALERO          | VALERO          | VALERO 1657              TUCSON       AZ 
VERIZON         | VERIZON         | VERIZON WRLS MYACCT VE   FOLSOM       CA 
VICTORIASSECRET | VICTORIASSECRET | VICTORIA'S SECRET 0020   FAIRFAX      VA 
"""

import csv
from .merchant_trie import generate_merchant_trie, standardize

MERCHANT_TRIE = generate_merchant_trie()

def find_merchant(transaction):
	"""
		Looks through a transaction for a merchant that matches a trie
	"""
	longest = None
	transaction = standardize(transaction)
	for start in range(len(transaction) - 1):
		if MERCHANT_TRIE.has_keys_with_prefix(transaction[start]):
			for end in range(start, len(transaction) + 1):
				if transaction[start:end] in MERCHANT_TRIE and \
				   len(transaction[start:end]) > 3 and \
				   (longest is None or len(longest) < len(transaction[start:end])):
					longest = transaction[start:end]
	return longest

def column_print(merchant, expected, transaction):
	"""print out infomation in a nice format"""
	out = ""
	out += merchant[:15] + (" " * (15 - len(merchant))) + " | "
	out += expected[:15] + (" " * (15 - len(expected))) + " | "
	return out + transaction

def main():
	"""runs the file"""
	# This file is located in S3 under
	# s3yodlee/development/bank/3_year_card_sample.txt
	# transactions = open("data/input/3_year_card_sample.log")
	transactions = open("data/input/merchant_labels.csv", encoding="ISO-8859-1")
	reader = csv.reader(transactions, delimiter=',')
	correct = 0
	count = 0
	guesses = 0
	print(column_print("Guessed", "Actual", "Transaction"))
	for line in reader:
		merchant = find_merchant(line[0])
		if merchant != None:
			guesses += 1
		if standardize(merchant) == standardize(line[1]):
			correct += 1
		print(column_print(standardize(merchant), standardize(line[1]), line[0]))
		count += 1
		if count > 1000:
			break
	print("guesses: ", guesses)
	print("correct: ", correct)
	print("incorrect: ", guesses - correct)
	print("total: ", count)

if __name__ == "__main__":
	main()
