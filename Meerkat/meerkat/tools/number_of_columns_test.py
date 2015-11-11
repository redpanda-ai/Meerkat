#!/usr/bin/python3.3
"""Prints the lines that have a certain number of columns."""
import sys
SIZES = {}
F = open(sys.argv[1])
LINE_COUNT = 0
for line in F:
	LINE_COUNT += 1
	fields = line.split("\t")
	size = len(fields)
	if size not in SIZES:
		SIZES[size] = 1
	else:
		SIZES[size] += 1
print(SIZES)
