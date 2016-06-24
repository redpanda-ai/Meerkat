from random import randrange

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

items = [1, 2, 3, 4, 5, 6, 7]
sample = reservoir_sampling(items, 5)
print(sample)
