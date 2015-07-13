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

from .merchant_trie import *
import time
import csv

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
				   (longest == None or len(longest) < len(transaction[start:end])):
					longest = transaction[start:end]

	return longest
def column_print(merchant, expected, transaction):
	out = ""
	out += merchant[:15] + (" " * (15 - len(merchant))) + " | "
	out += expected[:15] + (" " * (15 - len(expected))) + " | "
	return out + transaction

def main():
	"""runs the file"""
	# This file is located in S3 under s3yodlee/development/bank/3_year_card_sample.txt
	# transactions = open("data/input/3_year_card_sample.log")
	transactions = open("data/input/merchant_labels.csv",encoding = "ISO-8859-1")
	reader = csv.reader(transactions, delimiter=',')

	confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]

	print(column_print("Guessed", "Actual", "Transaction"))
	count = 0
	for line in reader:
		merchant = standardize(find_merchant(line[0]))
		actual = standardize(line[1])
		transaction = line[0]
		

		count += 1
		row, col = 0, 0

		if actual == "":
			col = 1

		if merchant == "":
			row = 2
		elif merchant != actual:
			row = 1


		confusion_matrix[row][col] += 1

		print(column_print(merchant, standardize(actual), transaction))

	print("\n")
	print("              found | null")
	print("got correct %7d | %5d" % (confusion_matrix[0][0], confusion_matrix[0][1]))
	print("got wrong %9d | %5d" %   (confusion_matrix[1][0], confusion_matrix[1][1]))
	print("got null %10d | %5d" %   (confusion_matrix[2][0], confusion_matrix[2][1]))
	print("total records:", count)


if __name__ == "__main__":
	main()
