from random import randrange
import pandas as pd

def reservoir_sampling(items, k):
	"""Reservoir Sampling"""
	sample = items[0:k]

	for i in range(k, len(items)):
		print("Looping")
		j = randrange(0, i+1)
		if j < k:
			print("I {0}".format(i))
			print("J {0}".format(j))
			sample[j] = items[i]

	return sample

"""
def chunky_sampling(input_file, k):

	first_chunk = True
	record_no = k
	chunksize = 6
	sample = None
	for chunk in pd.read_csv(input_file, chunksize=chunksize):
		if first_chunk:
			sample = chunk[0:k]
			chunk = chunk[k:]
			first_chunk = False
		else:
			chunk = pd.concat([
			#print("Sample\n{0}".format(sample))
		#print(chunk)
		row_iterator = chunk.iterrows()
		for i, row in row_iterator:
			print("I {0}".format(i))
			j = randrange(0, i)
			if j <= k:
				sample.iloc[j] = row
				print("Sample\n{0}".format(sample))
			print(row.record_no)
"""
items = [1, 2]
sample = reservoir_sampling(items, 1)
print(sample)

#chunky_sampling("meerkat/fat_head/input.tab", 2)
