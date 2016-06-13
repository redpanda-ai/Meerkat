"""A simple json schema validator, use schema.py for a more complete check"""

import json
import sys
from jsonschema import validate

#Usage:   python3.3 validator.py <schema_file> <example_file>
#Example: python3.3 validator.py schema_output.json example_output.json

def main_process():
	"""simple json schema validation"""
	schema_file, example_file = sys.argv[1], sys.argv[2]
	json_data = open(schema_file)
	example_data = open(example_file)

	schema = json.load(json_data)
	example = json.load(example_data)

	validate(example, schema)
	print("JSON is valid")

if __name__ == "__main__":
	main_process()
