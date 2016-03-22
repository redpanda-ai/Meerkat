"""Fixtures for test_various_tools"""

from queue import *

def get_queue():
	"""Return a queue"""
	non_empty_queue = Queue()
	non_empty_queue.put(1)
	non_empty_queue.put(2)
	non_empty_queue.put(3)

	empty_queue = Queue()

	return {
		"non_empty": non_empty_queue,
		"empty": empty_queue
}
