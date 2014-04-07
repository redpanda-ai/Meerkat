#!/bin/python3

import inspect
import sys
import re
import uuid
from longtail import bulk_loader

def start():
	try:
			input_file = open(sys.argv[1], encoding='utf-8')
			#params = json.loads(input_file.read())
			input_file.close()
	except IOError:
		logging.error(sys.argv[1] + " not found, aborting.")
		sys.exit()
	lines = get_file_lines(sys.argv[1])
	imports = get_imports(lines)
	process(lines, imports)

def get_file_lines(input_filename):
	with open(input_filename, 'r') as input_file:
		lines = [line.rstrip('\n') for line in input_file]
	return lines

def filter_symbols(symbols, imports):
	#black_list = {}
	black_list = { '__builtins__', '__cached__', '__doc__',
		'__file__', '__initializing__', '__loader__', '__name__', '__package__' }
	clear, masked = [], []
	for symbol in symbols:
		if symbol in black_list:
			#print("Found {0} in black_list".format(symbol))
			clear.append(symbol)
		elif symbol in imports:
			#print("Found {0} in imports".format(symbol))
			clear.append(symbol)
		else:
			#print("Adding {0} to masked".format(symbol))
			masked.append(symbol)
	return clear, masked

def inspect_methods():
	members = inspect.getmembers(bulk_loader.ThreadConsumer, predicate=inspect.isfunction)
	my_list = []
	for x, y in members:
		my_list.append(x)
	#print("Members:\n{0}".format(members))
	private_method_re = re.compile("^_(.*)(__.*[^_])$")
	public_method_re = re.compile("^([^_].*)$")
	private_methods, public_methods = [], []
	for item in my_list:
		if private_method_re.search(item):
			matches = private_method_re.match(item)
			private_methods.append(matches.group(2))
		elif public_method_re.search(item):
			matches = public_method_re.match(item)
			public_methods.append(matches.group(1))

	#print("Private methods:\n{0}".format(private_methods))
	#print("Public methods:\n{0}".format(public_methods))
	#TODO: public methods not specific enough
	return private_methods, public_methods


def get_imports(lines):
	from_import_re = re.compile("^from\s*(.*)\s*import(.*)")
	import_re = re.compile("^import\s*(.*)")
	results = {}
	for line in lines:
		if from_import_re.search(line):
			matches = from_import_re.match(line)
			symbols = matches.group(2).split(",")
			for s in symbols:
				s = s.strip()
				results[s] = ''
		elif import_re.search(line):
			matches = import_re.match(line)
			symbols = matches.group(1).split(",")
			for s in symbols:
				s = s.strip()
				results[s] = ''
	return results

def mask_symbols(lines, masked, private_methods):
	mask_dict, private_method_dict = {}, {}
	for item in masked:
		mask_dict[item] = "x_" + uuid.uuid5(uuid.NAMESPACE_DNS, item).hex
	for item in private_methods:
		private_method_dict[item] = "__x_" + uuid.uuid5(uuid.NAMESPACE_DNS, item).hex
	line_comment_re = re.compile("^\s*#.*")
	docstring_re = re.compile('^\s*""".*"""$')
	for line in lines:
		if line_comment_re.search(line):
			continue
		if docstring_re.search(line):
			continue
		for mask in masked:
			line = line.replace(mask, mask_dict[mask])
		for method in private_methods:
			line = line.replace(method, private_method_dict[method])
		print(line)
	#print(mask_dict)


def process(lines, imports):
	symbols = dir(bulk_loader)
	clear, masked = filter_symbols(symbols, imports)
	private_methods, _ = inspect_methods()
	mask_symbols(lines, masked, private_methods)

#	print("Symbols:\n{0}".format(symbols))
#	print("Clear:\n{0}".format(clear))
#	print("Masked:\n{0}".format(masked))

start()
