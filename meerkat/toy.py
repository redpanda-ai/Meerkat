import random
import queue
import logging
from time import sleep
import threading

def start_consumers(params, num_consumers):
	for	i in range(num_consumers):
		consumer = Consumer(i, params)
		consumer.setDaemon(True)
		consumer.start()

def start_producer(params, thread_id, num_elements):
	producer = Producer(thread_id, params, num_elements)
	producer.setDaemon(True)
	producer.start()
	return producer

class Producer(threading.Thread):
	def __init__(self, thread_id, params, num_elements):
		#global concurrency_queue
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.params = params
		self.num_elements = num_elements
		self.data_queue = params["data_queue"]
		concurrency_queue.put(self.thread_id)
		self.data_queue_populated = False
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
		#global concurrency_queue
		my_logger = logging.getLogger("thread {0}".format(self.thread_id))
		my_logger.setLevel(logging.INFO)
		my_formatter = logging.Formatter("%(asctime)s = %(name)s - %(levelname)s - %(message)s")
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.WARNING)
		console_handler.setFormatter(my_formatter)
		my_logger.addHandler(console_handler)
		params, self.data_queue, num_elements = self.params, self.data_queue, self.num_elements
		for i in range(num_elements):
			foo = random.randint(1, 500)
			self.data_queue.put(foo)
			my_logger.warning("Added {0}".format(foo))
			sleep(0.01)

		self.data_queue_populated = True
		concurrency_queue.get()
		concurrency_queue.task_done()

class Consumer(threading.Thread):
	def __init__(self, thread_id, params):
		#global concurrency_queue
		threading.Thread.__init__(self)
		self.thread_id = thread_id
		self.params = params
		self.data_queue = params["data_queue"]
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
		#global concurrency_queue
		my_logger = logging.getLogger("thread {0}".format(self.thread_id))
		my_logger.setLevel(logging.INFO)
		my_formatter = logging.Formatter("%(asctime)s = %(name)s - %(levelname)s - %(message)s")
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.WARNING)
		console_handler.setFormatter(my_formatter)
		my_logger.addHandler(console_handler)
		params, data_queue =  self.params, self.data_queue
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
concurrency_queue = queue.Queue()
my_params["data_queue"] = queue.Queue()
producer = start_producer(my_params, -1, 100)
start_consumers(my_params, 10)
my_params["data_queue"], my_params["data_queue_populated"] = producer.data_queue, producer.data_queue_populated
my_params["data_queue"].join()
concurrency_queue.join()
logging.critical("End of Program")

