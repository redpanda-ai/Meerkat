"""Finds merchants in given transaction strings

Created on June 30, 2015
@author: Sivan Mehta
"""

from .merchant_trie import *
import time
import csv

MERCHANT_TRIE = generate_merchant_trie()

def find_merchant(transaction):
	"""
		Looks through a transaction for a merchant that matches a trie
	"""
	transaction = standardize(transaction)
	for start in range(len(transaction) - 1):
		if MERCHANT_TRIE.has_keys_with_prefix(transaction[start]):
			for end in range(start + 1, len(transaction)):
				if transaction[start:end] in MERCHANT_TRIE and len(transaction[start:end]) > 3:
					return transaction[start:end]
		else:
			return None

def main():
	"""runs the file"""
	# This file is located in S3 under s3yodlee/development/bank/3_year_card_sample.txt
	transactions = open("data/input/3_year_card_sample.log")
	reader = csv.reader(transactions, delimiter='|')
	correct = 0
	count = 0
	guesses = 0
	for line in reader:
		merchant = find_merchant(line[0])
		if merchant != None:
			guesses += 1
		if standardize(merchant) == standardize(line[1]):
			correct += 1
		elif merchant != None:
			print(merchant, " | ", line[1], " | ", line[0])
		count += 1
		if count > 1000:
			break
	print("guesses: ", guesses)
	print("correct: ", correct)
	print("incorrect: ", guesses - correct)
	print("total: ", count)

if __name__ == "__main__":
	main()