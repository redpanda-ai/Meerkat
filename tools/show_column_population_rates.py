#!/usr/bin/python

import os, sys

def fill_dict(fields):
	columns, c2 = {}, {} 
	for i in range(len(fields)):
		columns[i] = fields[i]
		c2[i] = 0
	return columns, c2

F = open(sys.argv[1])
line_count = 0
columns, c2 = {}, {} 

fields = []
for line in F:
	fields = line.split("\t")
	if line_count == 0:
		columns, c2 = fill_dict(fields)
	else:
		for i in range(len(fields)):
			if fields[i].strip() != '':
				c2[i] += 1	
						
	line_count += 1


for key in sorted(c2.iterkeys()):
	#print str(key) + ": " + str(c2[key]) + " ( " + str(1.0 *c2[key]/line_count) + ") " + columns[key]
	print str(key) + ": " + columns[key] + " " + str(round(1.0 *c2[key]/line_count,2))


