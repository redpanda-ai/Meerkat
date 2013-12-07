#!/usr/bin/python

import os, sys

sizes = {}

F = open(sys.argv[1])
line_count = 0
for line in F:
	line_count += 1
	fields = line.split("\t")
	size = len(fields)
	if size not in sizes:
		sizes[size] = 1
	else:
		sizes[size] += 1
print sizes
