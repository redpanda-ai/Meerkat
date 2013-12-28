#!/usr/bin/python
"""Prints a report about the columns in a tab delimited file and
their population rates."""
import sys

def fill_dict(fields):
	"""Files the columns and columns_2 dictionaries."""
	columns, columns_2 = {}, {}
	for j in range(len(fields)):
		columns[j] = fields[j]
		columns_2[j] = 0
	return columns, columns_2

F = open(sys.argv[1])
LINE_COUNT = 0
COLUMNS, COLUMNS_2 = {}, {}

FIELDS = []
for line in F:
	FIELDS = line.split("\t")
	if LINE_COUNT == 0:
		COLUMNS, COLUMNS_2 = fill_dict(FIELDS)
	else:
		for i in range(len(FIELDS)):
			if FIELDS[i].strip() != '':
				COLUMNS_2[i] += 1
	LINE_COUNT += 1


for key in sorted(list(COLUMNS_2.keys())):
	print (str(key), ": ", COLUMNS[key], " "\
	, str(round(1.0 *COLUMNS_2[key]/LINE_COUNT,2)))


