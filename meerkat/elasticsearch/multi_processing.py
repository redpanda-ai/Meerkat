import os
import sys
import time
import pandas as pd
import multiprocessing as mp

kwargs = {
	'chunk_num': 1
}

def worker(df, **kwargs):
	print('This is pid: {}'.format(os.getpid()))
	print('This is chunk: {}'.format(kwargs['chunk_num']))

def single_processing(filename):
	start_time = time.time()
	reader = pd.read_csv(filename, chunksize=10000)

	chunk = 1
	for df in reader:
		kwargs['chunk_num'] = chunk
		worker(df, **kwargs)
		chunk += 1
	print('The number of seconds for single processing: {}'.format(time.time() - start_time))

def multi_processing(filename):
	start_time = time.time()
	reader = pd.read_csv(filename, chunksize=10000)
	pool = mp.Pool(mp.cpu_count())

	chunk = 1
	for df in reader:
		kwargs['chunk_num'] = chunk
		pool.apply_async(worker, [df], kwargs)
		chunk += 1
	print('The number of seconds for multi processing: {}'.format(time.time() - start_time))

if __name__ == '__main__':
	# single_processing('./selected-lists-5224.csv')
	multi_processing('./selected-lists-5224.csv')
