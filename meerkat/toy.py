import random
import queue
import logging
from time import sleep
import threading


def load_data_queue(params, num_elements):
	data_queue = queue.Queue()
	for i in range(num_elements):
		data_queue.put(random.randint(1, 500))
	data_queue_populated = True
	return data_queue, data_queue_populated

def start_consumers(params, num_consumers):
	for	i in range(num_consumers):
		consumer = Consumer(i, params)
		consumer.setDaemon(True)
		consumer.start()

class Consumer(threading.Thread):
	def __init__(self, thread_id, params):
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.params = params
		self.data_queue = params["data_queue"]
		self.params["concurrency_queue"].put(self.thread_id)
		self.__set_logger()

	def __set_logger(self):
		my_logger = logging.getLogger("thread {0}".format(self.thread_id))
		my_logger.setLevel(logging.INFO)
		my_formatter = logging.Formatter("%(asctime)s = %(name)s - %(levelname)s - %(message)s")
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.WARNING)
		console_handler.setFormatter(my_formatter)
		my_logger.addHandler(console_handler)
		my_logger.warning("Log initialized")

	def run(self):
		my_logger = logging.getLogger("thread {0}".format(self.thread_id))
		my_logger.setLevel(logging.INFO)
		my_formatter = logging.Formatter("%(asctime)s = %(name)s - %(levelname)s - %(message)s")
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.WARNING)
		console_handler.setFormatter(my_formatter)
		my_logger.addHandler(console_handler)
		params, data_queue =  self.params, self.data_queue
		concurrency_queue = params["concurrency_queue"]
		data_queue_populated = params["data_queue_populated"]
		while True:
			if data_queue_populated and data_queue.empty():
				concurrency_queue.get()
				concurrency_queue.task_done()
				my_logger.warning("Consumer finished, concurrency_queue size: {0}".format(concurrency_queue.qsize()))
				return
			sleep_time = data_queue.get()
			my_logger.warning("Data queue size was {1}, Sleeping for {0:03} ms".format(sleep_time, data_queue.qsize()))
			sleep(0.001 * sleep_time)
			data_queue.task_done()

#main program
my_params = {}
my_params["data_queue_populated"] = False
my_params["concurrency_queue"] = queue.Queue()
my_params["data_queue"], my_params["data_queue_populated"] = load_data_queue(my_params, 1000)
start_consumers(my_params, 10)
my_params["data_queue"].join()
my_params["concurrency_queue"].join()
logging.critical("End of Program")


