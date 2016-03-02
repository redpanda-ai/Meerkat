import json, sys
from jsonschema import validate

#Usage:   python3.3 validator.py <schema_file> <example_file>
#Example: python3.3 validator.py schema_output.json example_output.json

"""Main program"""
schema_file, example_file = sys.argv[1], sys.argv[2]
json_data = open(schema_file)
example_data = open(example_file)

schema = json.load(json_data)
example = json.load(example_data)

result = validate(example, schema)
print("JSON is valid")
