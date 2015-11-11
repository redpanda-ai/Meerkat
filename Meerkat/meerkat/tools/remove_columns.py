#!/usr/bin/python
"""Removes unwanted columns from tab delimited files."""
import os, sys

JUNK = [0, 3, 4, 16, 32, 33, 34, 45, 57, 61, 62, 63\
, 64, 65, 69, 70, 74, 76, 84, 102, 107, 108, 109, 110\
, 111, 112, 114, 115, 116, 117, 118]

COMMAND = "cat " + sys.argv[1] + " | awk ' { FS = \"\\t\"; print "
for i in range(119):
	if i not in JUNK:
		foo = " \"\\t\" "
		COMMAND += "$" + str(i) + foo
COMMAND = COMMAND[0:len(COMMAND)-len(foo)]
COMMAND += "} ' > " + sys.argv[2]
os.system(COMMAND)
