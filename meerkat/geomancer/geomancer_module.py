import logging
import logging.config
import yaml
import os
import sys

script_name = os.path.basename(sys.argv[0])
logging.config.dictConfig(yaml.load(open('meerkat/geomancer/logging.yaml', 'r')))
logger = logging.getLogger(script_name)

class GeomancerModule:
	name = "geomancer_module"
	"""Contains methods and data of Geomancer Module"""
	def __init__(self, common_config, config):
		"""Constructor"""
		self.common_config = common_config
		self.config = config
		logger.info(GeomancerModule.name)
		#for key in common_config:
		#	self.config[key] = common_config[key]

	def main_process(self):
		"""Execute the main program"""
		return self.common_config
