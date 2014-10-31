#!/usr/local/bin/python3.3

import json

import tornado.ioloop
from tornado_json.routes import get_routes
from tornado_json.application import Application

import meerkat.web_service as api

#################### USAGE ######################

# python3.3 -m meerkat.web_service

#################################################

def main():

	routes = get_routes(api)

	print(
		json.dumps([(url, repr(route)) for url, route in routes], indent=2)
	)

	# Create the application
	application = Application(routes=routes, settings={})

	# Start the application on port 8888
	application.listen(8888)
	tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()