import sys

#arg 1 input file
#arg 2 prefix
c = 10001
with open(sys.argv[1]) as input_file:
	for line in input_file:
		line = line.strip()
		if len(line) > 0:
			print("{0} {1}{2}".format(line.strip(), sys.argv[2], str(c)))
		c += 1
