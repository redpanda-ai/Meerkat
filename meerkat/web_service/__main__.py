#!/usr/local/bin/python3.3

import json

import tornado.ioloop
from tornado_json.routes import get_routes
from tornado_json.application import Application

import api

def main():

	routes = get_routes(api)

	# Create the application
    application = Application(routes=routes, settings={})

    # Start the application on port 8888
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()