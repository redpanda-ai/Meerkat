import csv
import sys
from collections import defaultdict, OrderedDict

def sort_the_file(my_file):
	print("Sorting {0}".format(my_file))
	my_header = defaultdict(list)
	my_map = defaultdict(list)
	count = 0
	with open(my_file, 'r') as file_one:
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
			if count < MAX_LINES:
				my_map[line].extend(row[0:len(row)])
				count += 1
			else:
				break
	return list(my_header.values())[0], OrderedDict(sorted(my_map.items(), key=lambda t: t[0]))

def diff(a, b):
	b = set(b)
	return [aa for aa in a if aa not in b]

def merge_the_files(map_a, map_b):
	print("starting merge")
	match_count, all_count = 0, 0
	entry_a, entry_b = None, None
	while map_a and map_b and all_count < 100005:
		if entry_a is None:
			entry_a = map_a.popitem(last=False)
			a = entry_a[0]
		if entry_b is None:
			entry_b = map_b.popitem(last=False)
			b = entry_b[0]
		if a == b:
			match_count += 1
			entry_a = None
			entry_b = None
		all_count += 1
		if all_count % 1000 == 0:
			print("All {0} Match {1}".format(all_count, match_count))
	print(match_count)
	print(a)
	print(b)



#Main program

SORT_KEYS = ["MEM_ID", "BANK_ACCOUNT_ID", "BANK_TRANSACTION_ID"]
MAX_LINES = sys.maxsize

file_1, file_2 = sys.argv[1], sys.argv[2]
header_1, map_1 = sort_the_file(file_1)
header_2, map_2 = sort_the_file(file_2)

remainder = diff(header_2, header_1)

merge_the_files(map_1, map_2)


print(header_1)
print(remainder)




