#!/usr/local/bin/python3.3

"""This module takes a sample of a particular merchant
and enables a human labeler to sort out erroneous 
transactions

Created on Jan 15, 2015
@author: J. Andrew Key
"""

#################### USAGE ##########################

# python3.3 -m meerkat.labeling_tools.labeling_prototype [config_file]
# python3.3 -m meerkat.labeling_tools.labeling_prototype config/labeling_prototype.json

#####################################################

import csv
import curses
import gzip
import json
import logging
import os
import sys

from meerkat.custom_exceptions import InvalidArguments

def usage():
	logging.error("Usage:\n"
		"python3.3 -m meerkat.tools.filter_s3_objetcs <path_to_configuration_file>")

def initialize():
	"""Validates the command line arguments."""
	input_file, params = None, None

	if len(sys.argv) != 2:
		usage()
		raise InvalidArguments(msg="Incorrect number of arguments", expr=None)

	try:
		with open(sys.argv[1], encoding='utf-8') as input_file:
			params = json.loads(input_file.read())
	except IOError:
		logging.error("%s not found, aborting.", sys.argv[1])
		sys.exit()
	return params

def start():
	my_file = PARAMS["sample_file"]
	display_cols = PARAMS["display_columns"]
	sort_cols = PARAMS["sort_columns"]
	logging.warning("Reading {0}".format(my_file))
	csv.field_size_limit(sys.maxsize)
	PARAMS["row_count"] = 0
	with gzip.open(my_file, 'rt') as file_one:
		csv_reader = csv.reader(file_one, delimiter='|')
		first_line = True
		for row in csv_reader:
			if first_line:
				disp_filter = [ idx for idx, x in enumerate(row) if x in display_cols ]
				disp_labels = [ row[idx] for idx, x in enumerate(row) if x in display_cols ]
				sort_filter = [ idx for idx, y in enumerate(row) if y in sort_cols ]
				PARAMS["display_labels"] = disp_labels
			disp_index = []
			for x in disp_filter:
				disp_index.append(row[x])
				PARAMS["display_index"] = disp_index
			sort_index = []
			for y in sort_filter:
				sort_index.append(row[y])
				PARAMS["sort_index"] = sort_index
				PARAMS["row_id"] = ".".join(sort_index)
			if first_line:
				logging.warning("Will display the following: {0}".format((disp_index)))
				PARAMS["header_index"] = disp_index
				first_line = False
				continue
			selection = custom_menu(row)
			if selection != 1:
				PARAMS["row_count"] += 1
			if selection == 2:
				curses.endwin()
				logging.critical("Saving file, have a nice day!")
				#TODO: Ship file to S3?
				sys.exit()
		curses.endwin()
		logging.critical("Wow you finished!")
		#TODO: Ship file to S3?
		sys.exit()

def display_value(screen, index, col_pos):
		screen.addstr(col_pos, 2, PARAMS["display_labels"][index], curses.A_BOLD)
		screen.addstr(col_pos + 1, 4, PARAMS["display_index"][index], curses.A_BOLD)
		return col_pos + 2

def custom_menu(row):
	screen = curses.initscr()
	curses.start_color()
	screen.clear()
	curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)
	curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
	curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
	curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
	screen.keypad(True)
	pos = 1
	x = None
	h = curses.color_pair(1)
	l = curses.color_pair(2)
	m = curses.color_pair(3)
	p = curses.color_pair(4)
	n = curses.A_NORMAL
	choices = ["Zero", "Skip", "Quit"]
	my_labels = PARAMS["labels"]
	for i in range(len(my_labels)):
		choices.append( my_labels[i]["name"])
	#choices.extend(PARAMS["labels"])
	row_base, col_base, menu_level = 7, 4, 1
	while x != ord('\n'):
		screen.clear()
		screen.border(0)
		screen.addstr(0, 2, "ROW_ID:   ", m)
		screen.addstr(0, 12, PARAMS["row_id"], l)
		screen.addstr(1, 2, "You have completed " + str(PARAMS["row_count"]) + " rows.", l)
		screen.addstr(2, 2, "Menu level: " + str(menu_level), curses.A_BOLD)
		screen.addstr(4, 2, "TRANSACTION TYPE", curses.A_BOLD)
		screen.addstr(5, 2, choices[pos], p)
		my_col = 7
		for i in range(len(PARAMS["display_labels"])):
			my_col = display_value(screen, i, my_col)

		row_base = my_col

		for i in range(1, len(choices)):
			if pos == i:
				screen.addstr(row_base + i, col_base, str(i) + " -> " + choices[i], h)
			else:
				screen.addstr(row_base + i, col_base, str(i) + " -> " + choices[i], n)

		screen.refresh()
		x = screen.getch()

		direct = False
		for i in range(1, len(choices)):
			if x == ord(str(i)[0]):
				pos = i
				direct = True
				break

		if not direct:
			if x == curses.KEY_RIGHT:
				if menu_level == 1:
					menu_level += 1
				else:
					curses.flash()
			elif x == curses.KEY_LEFT:
				if menu_level == 2:
					menu_level -= 1
				else:
					curses.flash()
			elif x == curses.KEY_DOWN:
				if pos < len(choices) - 1:
					pos += 1
				else:
					curses.flash()
			elif x == curses.KEY_UP:
				if pos > 1:
					pos -= 1
				else:
					curses.flash()
			elif x != ord('\n'):
				curses.flash()
			else:
				pass
	return pos

if __name__ == '__main__':
	PARAMS = initialize()
	start()
	x, y = custom_menu()
	curses.endwin()
	print("Your selection was {0}".format(x))

