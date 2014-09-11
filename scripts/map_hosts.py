import sys

c = 10001
with open(sys.argv[1]) as input_file:
	for line in input_file:
		line = line.strip()
		if len(line) > 0:
			print("{0} n{1}".format(line.strip(), str(c)))
		c += 1
