#!/usr/bin/python3.3

"""Finds the number of columns in each row."""
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
