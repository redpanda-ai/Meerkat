#!/usr/local/bin/python3.3
"""This module starts an HTTPS web service.
USAGE:
# python3.3 -m meerkat.web_service

EXAMPLE CURL COMMAND TO TEST WEB SERVICE:
# curl --insecure -s -X POST -d @big.json https://localhost:443/meerkat/ \
--header "Content-Type:application/json" | python3.3 -m json.tool

"""
#import json
import os

import tornado.httpserver
import tornado.ioloop
from tornado_json.application import Application
from tornado.options import define, options

from meerkat.web_service.api import Meerkat_API

import logging, yaml
from logging.handlers import TimedRotatingFileHandler


# Define Some Defaults
define("port", default=443, help="run on the given port", type=int)

def main():
	"""Launches an HTTPS web service."""

	# Log access to the web service
	tornado.options.parse_command_line()

	# Start the logs
	# create_timed_rotating_log("logs/web_service.log")
	logging.config.dictConfig(yaml.load(open('meerkat/web_service/logging.yaml', 'r')))

	# Define valid routes
	# pylint: disable=bad-continuation
	routes = [
		("/meerkat/v1.0.1/?", Meerkat_API),
		("/meerkat/v1.0.0/?", Meerkat_API),
		("/meerkat/?", Meerkat_API),
		("/status/index.html", Meerkat_API)
	]
	data_dir = "./"
	# Provide SSL key and certificate
	ssl_options = {
		"certfile" : os.path.join(data_dir, "server.crt"),
		"keyfile" : os.path.join(data_dir, "server.key"),
	}
	# Create the tornado_json.application
	application = Application(routes=routes, settings={})
	# Create the http server
	http_server = tornado.httpserver.HTTPServer(application,\
		ssl_options=ssl_options)

	# Start the http_server listening on default port
	http_server.listen(options.port)
	tornado.ioloop.IOLoop.instance().start()

#MAIN PROGRAM
if __name__ == '__main__':
	main()
