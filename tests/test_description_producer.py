'''Unit tests for meerkat.description_producer'''

from meerkat import description_producer
from meerkat.custom_exceptions import InvalidArguments, Misconfiguration
import unittest, queue, sys, socket, os, json

class TokenizeDescriptionTests(unittest.TestCase):

	"""Our UnitTest class."""

	config = """
{
	"concurrency" : 1,
	"input" : {
		"hyperparameters" : "config/hyperparameters/made_up_key_name.json",
		"filename" : "data/input/100_bank_transaction_descriptions.csv",
		"delimiter" : ",",
		"encoding" : "utf-8"
	},
	"logging" : {
		"level" : "critical",
		"path" : "logs/foo.log",
		"formatter" : "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
		"console" : true
	},
	"output" : {
		"results" : {
            "fields" : ["name", "chain_id", "category_labels", "address", "locality", "region", "postcode", "website", "tel", "chain_name", "latitude", "longitude", "neighborhood", "factual_id"],
            "labels" : ["BUSINESS_NAME", "STORE_ID", "FACTUAL_CATEGORY", "STREET", "CITY", "STATE", "ZIP_CODE", "WEBSITE", "PHONE_NUMBER", "CHAIN_NAME", "LATITUDE", "LONGITUDE", "NEIGHBORHOOD", "FACTUAL_ID"]
        },
		"file" : {
			"format" : "csv",
			"path" : "data/input/unittestDeletable.csv"
		}
	},
	"elasticsearch" : {
		"cluster_nodes" : ["s0:9200", "s1:9200", "s2:9200"
		, "s3:9200", "s4:9200", "s5:9200", "s6:9200"
		, "s7:9200", "s8:9200", "s9:9200", "s10:9200"
		, "s11:9200"],
		"index" : "new_index",
		"type" : "new_type",
		"subqueries" : {
			"largest_matching_string": {
				"field_boosts" : "standard_fields",
				"query_type" : "qs_query"
			}
		},
		"boost_labels" : [ "standard_fields", "composite.address" ],
		"boost_vectors" : {
			"factual_id" :        [ 0.0, 0.0 ],
			"name" :              [ 1.0, 0.0 ],
			"address" :           [ 1.0, 0.0 ],
			"address_extended" :  [ 1.0, 0.0 ],
			"po_box" :            [ 1.0, 0.0 ],
			"locality" :          [ 1.0, 0.0 ],
			"region" :            [ 1.0, 0.0 ],
			"post_town" :         [ 1.0, 0.0 ],
			"admin_region" :      [ 1.0, 0.0 ],
			"postcode" :          [ 1.0, 0.0 ],
			"country" :           [ 1.0, 0.0 ],
			"tel" :               [ 1.0, 0.0 ],
			"fax" :               [ 1.0, 0.0 ],
			"neighborhood" :      [ 1.0, 0.0 ],
			"email" :             [ 1.0, 0.0 ],
			"category_ids" :      [ 1.0, 0.0 ],
			"category_labels" :   [ 1.0, 0.0 ],
			"status" :            [ 1.0, 0.0 ],
			"chain_name" :        [ 1.0, 0.0 ],
			"chain_id" :          [ 1.0, 0.0 ],
			"pin.location" :      [ 0.0, 0.0 ],
			"composite.address" : [ 0.0, 3.0 ]
		},
		"composite_fields" : [
			{
				"address" : {
					"components" : [ "address", "address_extended", "locality", "region",
						"postcode", "country" ],
					"format" : "{0} {1} {2}, {3} {4}, {5}",
					"index" : "analyzed",
					"type" : "string"
				}
			}
		],
		"type_mapping" : {
			"mappings" : {
				"factual_type" : {
					"_source" : {
						"enabled" : true
					},
				"properties" : {
					"factual_id" : { "index" : "analyzed", "type" : "string" },
					"name" : { "index" : "analyzed", "type" : "string" },
					"address" : { "index" : "analyzed", "type" : "string" },
					"address_extended" : { "index" : "analyzed", "type" : "string" },
					"po_box" : { "index" : "analyzed", "type" : "string" },
					"locality" : { "index" : "analyzed", "type" : "string" },
					"region" : { "index" : "analyzed", "type" : "string" },
					"post_town" : { "index" : "analyzed", "type" : "string" },
					"admin_region" : { "index" : "analyzed", "type" : "string" },
					"postcode" : { "index" : "analyzed", "type" : "string" },
					"country" : { "index" : "analyzed", "type" : "string" },
					"tel" : { "index" : "analyzed", "type" : "string" },
					"fax" : { "index" : "analyzed", "type" : "string" },
					"neighborhood" : { "index" : "analyzed", "type" : "string" },
					"website" : { "index" : "analyzed", "type" : "string" },
					"email" : { "index" : "analyzed", "type" : "string" },
					"category_ids" : { "index" : "analyzed", "type" : "string" },
					"category_labels" : { "index" : "analyzed", "type" : "string" },
					"status" : { "index" : "analyzed", "type" : "string" },
					"chain_name" : { "index" : "analyzed", "type" : "string" },
					"chain_id" : { "index" : "analyzed", "type" : "string" },
					"pin" : { "properties" : { "location" : {
						"type" : "geo_shape", "tree" : "quadtree", "precision" : "1m" } }
					}
				}
			}
		},
		"settings" : {
			"number_of_replicas" : 1,
			"number_of_shards" : 12
		}
		}
	}
}"""


	def setUp(self):
		self.params = json.loads(self.config)
		self.desc_queue, self.result_queue = queue.Queue(), queue.Queue()
		for arg in sys.argv[1:]:
			sys.argv.remove(arg)

	def test_usage(self):
		"""The point of this function is to print usage information to the user"""
		result = description_producer.usage()
		self.assertEqual("Usage:\n\t<path_to_json_format_config_file>", result)

	def test_get_desc_queue_returns_queue(self):
		"""Ensure returns an instance of Queue"""
		my_queue, non_physical = description_producer.get_desc_queue(self.params)
		self.assertTrue(isinstance(my_queue, queue.Queue))

	def test_get_desc_queue_is_not_empty(self):
		"""Ensure queue is not empty"""
		my_queue, non_physical = description_producer.get_desc_queue(self.params)
		self.assertFalse(my_queue.empty())

	def test_initialize_no_file_name(self):
		"""Config file not provided"""
		self.assertRaises(InvalidArguments, description_producer.initialize)

	def test_initialize_file_does_not_exist(self):
		"""Config file doesn't exist"""
		sys.argv.append("data/somethingThatWontExist.csv")
		self.assertRaises(SystemExit, description_producer.initialize)

	def test_initialize_too_many_arguments(self):
		"""Too Many Options"""
		sys.argv.append("data/somethingThatWontExist.csv")
		sys.argv.append("argument")
		self.assertRaises(InvalidArguments, description_producer.initialize)

	def test_tokenize(self):
		"""The point of this function is to start a number of
		consumers as well as a starting queue and a result queue.
		At the end a call to write_output_to_file should be made"""

	def test_write_output_to_file_writes_file(self):
		"""Ensure actually writes a file"""
		self.result_queue.put({"PERSISTENTRECORDID":"123456789"})
		description_producer.write_output_to_file(self.params, self.result_queue, [])
		self.assertTrue(os.path.isfile("data/input/unittestDeletable.csv"))
		os.remove("data/input/unittestDeletable.csv")

	def test_write_output_to_file_empties_queue(self):
		"""Ensure queue is empty at end"""
		self.result_queue.put({"PERSISTENTRECORDID":"123456789"})
		description_producer.write_output_to_file(self.params, self.result_queue, [])
		self.assertTrue(self.result_queue.empty())
		os.remove("data/input/unittestDeletable.csv")

	def test_validate_logging(self):
		"""Ensure 'logging' key is in configuration"""
		del self.params["logging"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_logging_path(self):
		"""Ensure 'logging.path' key is in configuration"""
		del self.params["logging"]["path"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_elasticsearch_index(self):
		"""Ensure 'elasticsearch.index' key is in configuration"""
		del self.params["elasticsearch"]['index']
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_elasticsearch_type(self):
		"""Ensure 'elasticsearch.type' key is in configuration"""
		del self.params["elasticsearch"]['type']
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_elasticsearch(self):
		"""Ensure 'elasticsearch' key is in configuration"""
		del self.params["elasticsearch"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_empty_config(self):
		"""Ensure configuration is not empty"""
		self.params = {}
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_missing_concurrency(self):
		"""Ensure 'concurrency' key is in configuration"""
		del self.params["concurrency"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_positive_concurrency(self):
		"""Ensure 'concurrency' value is a positive integer"""
		self.params["concurrency"] = 0
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_input_key(self):
		"""Ensure 'input' key is in configuration"""
		del self.params["input"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_input_file(self):
		"""Ensure input file is provided"""
		del self.params["input"]["filename"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_validate_encoding(self):
		"""Ensure encoding key is in configuration"""
		del self.params["input"]["encoding"]
		self.assertRaises(Misconfiguration, description_producer.validate_params, self.params)

	def test_hyperparameters_default(self):
		"""Ensure parameter key defaults to default.json"""
		del self.params["input"]["hyperparameters"]
		description_producer.validate_params(self.params)
		self.assertEqual(self.params["input"]["hyperparameters"], "config/hyperparameters/default.json")

	def test_false_key_throws_error(self):
		"""Ensure not existent key throws error"""
		self.assertRaises(SystemExit, description_producer.load_hyperparameters, self.params)				

if __name__ == '__main__':
	unittest.main(argv=[sys.argv[0]])
