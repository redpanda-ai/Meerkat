#!/usr/bin/local/python3
"""generates a report of columns and their population rates."""
import sys

def fill_dict(fields):
	columns, c2 = {}, {} 
	for i in range(len(fields)):
		columns[i] = fields[i]
		c2[i] = 0
	return columns, c2

F = open(sys.argv[1])
line_count = 0
columns, c2 = {}, {} 

FIELDS = []
for line in F:
	FIELDS = line.split("\t")
	if line_count == 0:
		columns, c2 = fill_dict(FIELDS)
	else:
		for i in range(len(FIELDS)):
			if FIELDS[i].strip() != '':
				c2[i] += 1
	line_count += 1

for key in sorted(c2.keys()):
	print (str(key).strip(), "\t", columns[key].strip(), "\t"\
	, str(round(1.0 *c2[key]/line_count,2)).strip())
