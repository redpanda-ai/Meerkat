import csv
import gzip
import sys
import logging

from collections import defaultdict, OrderedDict

def sort_the_file(my_file):
	logging.warning("Reading {0}".format(my_file))
	my_header = defaultdict(list)
	my_map = defaultdict(list)
	count = 0
	tock = 20000
	with gzip.open(my_file, 'rt') as file_one:
		csv_reader = csv.reader(file_one, delimiter='|')
		first_line = True
		for row in csv_reader:
			if first_line:
				my_filter = [ idx for idx, x in enumerate(row) if x in SORT_KEYS ]
			bar = []
			for x in my_filter:
				bar.append(row[x])
			line = ".".join(bar)
			if first_line:
				my_header[line].extend(row[0:len(row)])
				first_line = False
				continue
			if count % tock == 0:
				sys.stdout.write('.')
				sys.stdout.flush()
			if count < MAX_LINES:
				my_map[line].extend(row[0:len(row)])
				count += 1
			else:
				break
	logging.warning("Sorting")
	return list(my_header.values())[0], OrderedDict(sorted(my_map.items(), key=lambda t: t[0])), count

def diff(a, b):
	b = set(b)
	return [aa for aa in a if aa not in b]

def get_columns(my_data, my_header, my_filter):
	new_filter = [ idx for idx, x in enumerate(my_header) if x in my_filter ]
	return [ my_data[1][x] for x in new_filter ]

def merge_the_files(expected_lines):
	logging.warning("Merging")
	match_count, all_count, tick = 0, 0, 10000
	tick = int(expected_lines / 20)
	entry_a, entry_b = None, None
	with gzip.open(merged_file, 'wt') as f_out:
		header_line = "|".join(header_1) + "|".join(remainder) + "\n"
		f_out.write(header_line)
		while map_1 and map_2 and all_count < expected_lines:
			if entry_a is None:
				entry_a = map_1.popitem(last=False)
				a = entry_a[0]
			if entry_b is None:
				entry_b = map_2.popitem(last=False)
				b = entry_b[0]
			if a == b:
				match_count += 1
				part_a = get_columns(entry_a, header_1, header_1)
				part_b = get_columns(entry_b, header_2, remainder)
				line = "|".join(part_a) + "|".join(part_b) + "\n"
				f_out.write(line)
				entry_a = None
				entry_b = None
			all_count += 1
			if all_count % tick == 0:
				sys.stdout.write('.')
				sys.stdout.flush()
	logging.warning(" Done!")

#Main program

SORT_KEYS = ["MEM_ID", "BANK_ACCOUNT_ID", "BANK_TRANSACTION_ID"]
MAX_LINES = sys.maxsize

file_1, file_2, merged_file = sys.argv[1], sys.argv[2], sys.argv[3]
header_1, map_1, count_1 = sort_the_file(file_1)
logging.warning("There were {0} records in the file.".format(count_1))
header_2, map_2, count_2 = sort_the_file(file_2)
logging.warning("There were {0} records in the file.".format(count_2))
#Abort if files have a different number of records
if count_1 != count_2:
	logging.critical("ERROR! Mismatched number of lines, aborting.")
	sys.exit()

logging.warning("Files have the same number of records, proceeding")
remainder = diff(header_2, header_1)
merge_the_files(count_1)

