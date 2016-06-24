from random import randrange
import pandas as pd

def reservoir_sampling(items, k):
	"""Reservoir Sampling"""
	sample = items[0:k]

	for i in range(k, len(items)):
		print("Looping")
		j = randrange(0, i + 1)
		if j < k:
			print("I {0}".format(i))
			print("J {0}".format(j))
			sample[j] = items[i]

	return sample

def df_reservoir_sampling(input_file, k, chunksize):
	first_chunk = True
	sample = None
	last_i = 0
	for chunk in pd.read_csv(input_file, chunksize=chunksize):
		print("Last I :{0}".format(last_i))
		if first_chunk:
			sample = chunk[0:k]
			chunk = chunk[k:]
			first_chunk = False

		print("Sample:\n{0}".format(sample))
		print("Chunk:\n{0}".format(chunk))

		row_iterator = chunk.iterrows()
		for i, row in row_iterator:
			print("I {0}".format(i))
			j = randrange(0, last_i + i + 1)
			if j < k:
				sample.iloc[j] = row
			this_i = i
		last_i += this_i + 1
	return sample

sample = df_reservoir_sampling("meerkat/fat_head/input.tab", 2, 6)
print("Final Sample:\n{0}".format(sample))

