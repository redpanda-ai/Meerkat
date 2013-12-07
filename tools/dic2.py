#!/usr/bin/python

import os, sys

def fill_dict(fields):
	columns, c2 = {}, {} 
	for i in range(len(fields)):
		columns[i] = fields[i]
		c2[i] = 0
	return columns, c2

F = open(sys.argv[1])
line_max = int(sys.argv[2])
line_count = 0
columns, c2 = {}, {} 

sizes = {}

d =  {}
for line in F:
	fields = line.split("\t")
	if line_count == 0:
		columns, c2 = fill_dict(fields)
	if line_count >= line_max:
		break	
	else:
		for i in range(len(fields)):
			cell = str(fields[i]).strip()
			sz = len(cell)
			if sz < 2:
				continue
			if sz not in d:
				d[sz] = {}
			if cell not in d[sz]:
				d[sz][cell] = 0 
			d[sz][cell] += 1

	line_count += 1

#print fields
for key in sorted(d.iterkeys()):
	print str(key) + ": " + str(d[key])

