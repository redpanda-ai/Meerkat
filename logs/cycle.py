import time, os

if os.path.isfile("logs/day1.log"):
	os.remove("logs/day1.log")

for path in os.listdir("logs"):
	if os.path.isfile("logs/" + path):
		path_number = int(path[-5])
		new_file = open("logs/.tmp/day%d.log" % (path_number - 1), "w+")
		with open("logs/" + path) as old_file:
			new_file.write(old_file.read())
