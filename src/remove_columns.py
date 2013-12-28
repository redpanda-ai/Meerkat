#!/usr/bin/python
"""Strips out useless columns."""
import os, sys

JUNK = [0, 3, 4, 16, 32, 33, 34, 45, 57, 61, 62, 63, 64\
, 65, 69, 70, 74, 76, 84, 102, 107, 108, 109, 110, 111\
, 112, 114, 115, 116, 117, 118]

SHELL = "cat " + sys.argv[1] + " | awk ' { FS = \"\\t\"; print "
for i in range(119):
	if i not in JUNK:
		foo = " \"\\t\" "
		SHELL += "$" + str(i) + foo
SHELL = SHELL[0:len(SHELL)-len(foo)]

SHELL += "} ' > " + sys.argv[2]
#print x
os.system(SHELL)
