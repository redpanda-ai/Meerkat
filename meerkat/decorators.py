"""This module demonstrates decorators."""
import time
from functools import wraps

def timethis(func):
	"""Decorator that reports execution time."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		"""Wrapper for timethis"""
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(func.__name__, end-start)
		return result
	return wrapper

def list_arguments(func):
	"""This decorator lists args and kwargs."""
	@wraps(func)
	def wrapper(*args, **kwargs):
		"""Another thing."""
		print("Something for before.")
		print("Args")
		for arg in args:
			print(arg)
		print("Kwargs")
		for key in kwargs:
			print("%s, %s" % (key, kwargs[key]))
		result = func(*args, **kwargs)
		print("Something for after.")
		return result
	return wrapper

def arguments(keyword=None, positional=None):
	"""Decorator that takes arguments"""
	if keyword is None:
		keyword = []
	if positional is None:
		positional = []
	def decorate(func):
		"""Innder decorator that accepts a function."""
		@wraps(func)
		def wrapper(*args, **kwargs):
			"""A wrapper."""
			#Check keyword arguments
			for key in keyword:
				if key not in kwargs:
					raise Exception("Missing %s" % (key))
			#Check positional arguments
			len_expected, len_actual = len(positional), len(args)
			if len_actual != len_expected:
				raise Exception("Expecting %d arguments but found %d arguments"\
					% (len_expected, len_actual))
			positional_counter = 0
			for name in positional:
				kwargs[name] = args[positional_counter]
				positional_counter += 1
			return func(*args, **kwargs)
		return wrapper
	return decorate

#@list_arguments
@timethis
@list_arguments
@arguments(keyword=["a", "b"], positional=["c"])
def boo(*args, **kwargs):
	"""This function uses the 'arguments' decorator"""
	print("a is %s" % (kwargs["a"]))
	print("b is %s" % (kwargs["b"]))
	for arg in args:
		print(arg)
	print("KWARGS %s" % (kwargs))

boo(1, a=1, b=2)

@arguments(keyword=["a", "b"])
def doo(*args, **kwargs):
	"""This function also uses the 'arguments' decorator"""
	print("a is %s" % (kwargs["a"]))
	print("b is %s" % (kwargs["b"]))
	for arg in args:
		print(arg)

doo(a=3, b=4)

@arguments()
def goo():
	"""This function also uses the 'arguments' decorator"""
	print("Goo doesn't do much.")

goo()
