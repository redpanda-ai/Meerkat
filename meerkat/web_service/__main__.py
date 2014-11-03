#!/usr/local/bin/python3.3

import json

from pprint import pprint
import tornado.ioloop
from tornado_json.routes import get_routes
from tornado_json.application import Application

from meerkat.web_service.api import Meerkat_API

#################### USAGE ######################

# python3.3 -m meerkat.web_service

# curl -X POST -d @example_input.json http://localhost:8888/web_service/api/meerkat/ --header "Content-Type:application/json"

#################################################

def main():

	routes = [("/meerkat/?", Meerkat_API)]

	# Create the application
	application = Application(routes=routes, settings={})

	# Start the application on port 8888
	application.listen(8888)
	tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()