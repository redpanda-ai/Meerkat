import sys
import os
import json
import csv
import time
import scipy.stats

def do_query(city_state):
	return """
	curl -s localhost:9200/factual_index/_search_shards?routing=%s -d '
	{
		"query":
		{
			"match" :
			{
				"city_state" : "%s"
			}
		}
	} ' > out.log
	""" % (city_state, city_state)


def main():
	observed = [0] * 10
	def count_shards():
		output = json.load(open("out.log", "r"))
		for seen in output["shards"]:
			shard = seen[0]
			observed[shard["shard"]] += 1

	seen_rows = 0
	start_time = time.time()

	with open("data/input/factual_100000.log", "r") as d:
		data = csv.reader(d, delimiter = "\t")
		for line in data:
			city_state = line[5] + ", " + line[6]
			# print(city_state)
			# if len(city_state) == 5: # some don't come in formatted correctly
			os.system(do_query(city_state))
			count_shards()
			sys.stdout.flush()
			sys.stdout.write("\r%d seen afer %2.2fs... " % (seen_rows, time.time() - start_time))
			seen_rows += 1

			if seen_rows > 1000:
				break

	print("done!")
	
	expected = [seen_rows/10] * 10
	print(observed)
	print(expected)
	certainty = scipy.stats.chisquare(observed, expected)[1]
	print("They are similar with a %2.2f%% certainty" % (certainty*100.0))

if __name__ == "__main__":
	main()
