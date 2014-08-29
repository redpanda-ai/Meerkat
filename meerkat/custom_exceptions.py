#!/usr/local/bin/python3.3

"""This is where we keeping custom exception classes for use within
multiple scripts.

Created on Dec 21, 2013
@author: J. Andrew Key
"""

class FileProblem(Exception):
	"""Wraps exceptions pertaining to failures to open files."""
	def __init__(self, expr, msg):
		super(FileProblem, self).__init__(msg)
		self.expr = expr
		self.msg = msg

class InvalidArguments(Exception):
	"""Exception raised for invalid command line arguments."""
	def __init__(self, expr, msg):
		super(InvalidArguments, self).__init__(msg)
		self.expr = expr
		self.msg = msg

class InvalidNumberOfLines(Exception):
	"""Wraps an exeception for passing in a non-integer as the number
	of lines to parse."""
	def __init__(self, expr, msg):
		super(InvalidNumberOfLines, self).__init__(msg)
		self.expr = expr
		self.msg = msg

class Misconfiguration(Exception):
	"""Wraps an exeception for handling bad configuration files."""
	def __init__(self, expr, msg):
		super(Misconfiguration, self).__init__(msg)
		self.expr = expr
		self.msg = msg

class UnsupportedQueryType(Exception):
	"""Wraps exceptions related to using an unsupported query."""
	def __init__(self, expr, msg):
		super(UnsupportedQueryType, self).__init__(msg)
		self.expr = expr
		self.msg = msg

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	print("This module is a library that contains useful functions; it should not be run from the console.")