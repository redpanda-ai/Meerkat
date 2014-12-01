#!/usr/local/bin/python3.3

import json
import os

from pprint import pprint

import tornado.httpserver
import tornado.ioloop

from tornado_json.routes import get_routes
from tornado_json.application import Application
from tornado.options import define, options

from meerkat.web_service.api import Meerkat_API
#Define some defaults
define("port", default=443, help="run on the given port", type=int)

#################### USAGE ######################

# python3.3 -m meerkat.web_service

# curl --insecure -s -X POST -d @big.json https://localhost:443/meerkat/ --header "Content-Type:application/json" | python3.3 -m json.tool
#################################################

def main():
	#Log some useful stuff
	tornado.options.parse_command_line()
	routes = [
		("/meerkat/?", Meerkat_API),
		("/status/index.html", Meerkat_API)
	]

	data_dir = "./"
	ssl_options = {
		"certfile" : os.path.join(data_dir, "server.crt"),
		"keyfile" : os.path.join(data_dir, "server.key"),
	}
	# Create the tornado_json.application
	application = Application(routes=routes, settings={})
	# Create the http server
	http_server = tornado.httpserver.HTTPServer(application, ssl_options=ssl_options)
	# Start the http_server listening on port 443
	http_server.listen(options.port)
	tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()
