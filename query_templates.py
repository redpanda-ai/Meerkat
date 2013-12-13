#!/usr/bin/python

def unpack_json_id(json_id,stuff):
	json_split = json_id.split(".")
	key = json_split[0]
	if len(json_split) == 1:
		return '{ "' + key + '" : { ' + unpack_attributes(stuff)[:-1] + " } } "
	value = ".".join(json_split[1:])
	return '{ "' + key + '" : ' + unpack_json_id(value,stuff) + " }"

def unpack_tuple_as_keyval(attribute):
	name, value = attribute[0:2]
	return '"' + name + '" : ' + str(value) + ","

def unpack_attributes(list_of_attributes):
	strings = []
	for attribute in list_of_attributes:
		name, value = attribute[0:2]
		strings.append('"' + name + '" : ' + str(value) + ",")
	return "\t" + "\n\t".join(strings)

STANDARD_QUERY = """
{
__attributes
	"fields" : [ 
		"BUSINESSSTANDARDNAME",
		"HOUSE",
		"STREET",
		"STRTYPE",
		"CITYNAME",
		"STATE",
		"ZIP",
		"pin.location" ],
	"query": {
		"bool" : {
			"should": [
				__query
			],
			"minimum_number_should_match" : 1,
			"boost" : 1.0
		}
	}
}
"""

BOOL_QUERY = """
{
__attributes
	"fields" : [ 
		"BUSINESSSTANDARDNAME",
		"HOUSE",
		"STREET",
		"STRTYPE",
		"CITYNAME",
		"STATE",
		"ZIP",
		"pin.location" ],
	"query": {
		"bool" : {
			__should,
			"minimum_number_should_match" : 1,
			"boost" : 1.0
		}
	}
}
"""
