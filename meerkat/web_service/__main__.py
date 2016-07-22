#!/usr/local/bin/python3.3
"""This module starts an HTTPS web service.
USAGE:
# sudo python3 -m meerkat.web_service

EXAMPLE CURL COMMAND TO TEST WEB SERVICE:
# curl --insecure -s -X POST -d @big.json https://localhost:443/meerkat/ \
--header "Content-Type:application/json" | python3.3 -m json.tool

@author: J. Andrew Key
@author: Sivan Mehta

"""
import logging
import os
import sys
import tornado.httpserver
import tornado.ioloop
import yaml

from tornado_json.application import Application
from tornado.options import define, options
from meerkat.web_service.api import Meerkat_API

# Define Some Defaults
define("port", default=(len(sys.argv) > 1 and str(sys.argv[1]) or 443),
	help="run on the given port", type=int)

def main():
	"""Launches an HTTPS web service."""

	# Log access to the web service
	tornado.options.parse_command_line()

	# Start the logs, configured by the following file
	logging.config.dictConfig(yaml.load( \
		open('meerkat/web_service/logging.yaml', 'r')))

	# Define valid routes
	routes = [
		("/meerkat/v2.4/?", Meerkat_API),
		("/meerkat/v2.3/?", Meerkat_API),
		("/meerkat/v2.2/?", Meerkat_API),
		("/meerkat/v2.1/?", Meerkat_API),
		("/meerkat/v2.0/?", Meerkat_API),
		("/meerkat/v1.9/?", Meerkat_API),
		("/meerkat/v1.8/?", Meerkat_API),
		("/meerkat/v1.7/?", Meerkat_API),
		("/meerkat/v1.6/?", Meerkat_API),
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
