import gzip
import sys
import math
import logging

from plumbum import local
from plumbum.cmd import wc

def write_lines_to_file(params):
	"""Write a list of lines to a gzipped file, named appropriately"""
	file_base = params["input_filename"][:-3]
	numerator = format(params["chunk_number"], params["leading_zeroes"])
	denominator = format(params["denominator"], params["leading_zeroes"])
	output_filename = "{0}.{1}.{2}.gz".format(file_base, numerator, denominator)
	print("Writing {0}".format(output_filename))
	with gzip.open(output_filename, "wt") as gzipped_output:
		for line in params["lines"]:
			gzipped_output.write(line + "\n")

def get_line_count(input_filename):
	"""Count the lines in a single file using pigz"""
	command = local["pigz"]["-c"]["-d"][input_filename] | wc["-l"]
	line_count = int(command()) - 1
	return line_count

def split_gzipped_file(input_filename, chunk_size, line_count):
	"""Split a single file zipped files into numbered pieces"""
	first_line = True
	denominator = int(round(line_count / chunk_size, 0)) -1
	params = {
		"chunk_number": 0, "chunk_size": chunk_size,
		"line_count": line_count, "input_filename": input_filename,
		"lines": [], "denominator": denominator,
		"leading_zeroes": "0" + str(int(math.log10(denominator) + 1))
	}
	with gzip.open(input_filename, "rt") as gzipped_input:
		#Go through each line in the input
		for line in gzipped_input:
			line = line.strip()
			#Capture the header once
			if first_line:
				header = line
				params["lines"] = [header]
				first_line = False
				continue
			else:
				params["lines"].append(line)
				#Dump a split to a file
				if len(params["lines"]) == params["chunk_size"] + 1:
					write_lines_to_file(params)
					params["chunk_number"] += 1
					params["lines"] = [header]
	#Write any tail split to a file as well
	if len(params["lines"]) > 1:
		write_lines_to_file(params)

def main():
	"""runs the file"""
	input_filename, chunk_size = sys.argv[1], int(sys.argv[2])
	line_count = get_line_count(input_filename)
	logging.warning("{0} contains {1} lines".format(input_filename, line_count))
	split_gzipped_file(input_filename, chunk_size, line_count)

main()