"""A simple json schema validator, use schema.py for a more complete check"""

import sys
from meerkat.various_tools import validate_configuration

#Usage:   python3.3 validator.py <schema_file> <example_file>
#Example: python3.3 validator.py schema_output.json example_output.json

if __name__ == "__main__":
	_ = validate_configuration(sys.argv[2], sys.argv[1])
