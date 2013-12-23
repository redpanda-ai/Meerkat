#!/usr/bin/python

"""This is where we keeping custom exception classes for use within multiple
scripts."""

class InvalidArguments(Exception):
	"""Exception raised for invalid command line arguments."""
	def __init__(self, expr, msg):
		self.expr = expr
		self.msg = msg	

class InvalidNumberOfLines(Exception):
	"""Wraps an exeception for passing in a non-integer as the number
	of lines to parse."""
	def __init__(self, expr, msg):
		self.expr = expr
		self.msg = msg	

class FileProblem(Exception):
	"""Wraps exceptions pertaining to failures to open files."""
	def __init__(self, expr, msg):
		self.expr = expr
		self.msg = msg	

class UnsupportedQueryType(Exception):
	"""Wraps exceptions related to using an unsupported query."""
	def __init__(self, expr, msg):
		self.expr = expr
		self.msg = msg	

