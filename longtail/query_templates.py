#!/usr/bin/python3.3

"""Words we wish to ignore while searching."""
STOP_WORDS = ["CHECK", "CARD", "CHECKCARD", "PAYPOINT", "PURCHASE", "LLC"]

#RESULT_FIELDS = ["BUSINESSSTANDARDNAME", "HOUSE"\
#, "PREDIR", "STREET", "STRTYPE", "CITYNAME", "STATE", "ZIP", "pin.location"]

#Template for building a search.
GENERIC_ELASTICSEARCH_QUERY = {}
GENERIC_ELASTICSEARCH_QUERY["from"] = 0
GENERIC_ELASTICSEARCH_QUERY["size"] = 10
#GENERIC_ELASTICSEARCH_QUERY["fields"] = RESULT_FIELDS
GENERIC_ELASTICSEARCH_QUERY["query"] = {}
GENERIC_ELASTICSEARCH_QUERY["query"]["bool"] = {}
GENERIC_ELASTICSEARCH_QUERY["query"]["bool"]["minimum_number_should_match"] = 1
GENERIC_ELASTICSEARCH_QUERY["query"]["bool"]["should"] = []

#Structure for composite_data_types.
COMPOSITES = {}
COMPOSITES[("address", " ")] = \
["HOUSE", "PREDIR", "STREET", "STRTYPE", "POSTDIR", "APTTYPE", "APTNBR"\
, "CITYNAME", "STATE"]
COMPOSITES[("phone", "")] = \
["AREACODE", "EXCHANGE", "PHONENUMBER"]

#Aliases for magic numbers.
NULL, FLOAT, DATE, INT, STRING = 0, 1, 2, 3, 4
DATA_TYPE_NAME, PATTERN = 0, 1
NAME, DATA_TYPE, INDEX = 0, 1, 2
KEY_NAME, KEY_DELIMITER = 0, 1


def get_composites(record_obj):
	"""Builds all composite features and adds them to the record object."""
	record_obj["composite"] = {}
	for key in COMPOSITES:
		components = COMPOSITES[key]
		key_name, key_delimiter = key[0:len(key)]
		row_components = [record_obj[component] \
		for component in components if component in record_obj]
		value_candidate = key_delimiter.join(row_components)
		if value_candidate != "":
			record_obj["composite"][key_name] = value_candidate

def get_create_object(es_index, es_type, cell_id):
	"""Builds object used for building a bulk create command."""
	create_object = {}
	create_object["create"] = {}
	create_object["create"]["_index"] = es_index
	create_object["create"]["_type"] = es_type
	create_object["create"]["_id"] = cell_id
	return create_object

def get_match_query(term, feature_name, boost):
	"""Returns a "match" style ElasticSearch query object"""
	match_query = {}
	match_query["match"] = {}
	match_query["match"][feature_name] = {}
	match_query["match"][feature_name]["query"] = term
	match_query["match"][feature_name]["type"] = "phrase"
	match_query["match"][feature_name]["boost"] = boost
	return match_query

def get_mapping_template(es_type_name, shards, replicas, column_meta\
, data_types):
	"Builds a mapping for an index type."""
	map_object = {}
	map_object["settings"] = {}
	map_object["settings"]["number_of_shards"] = shards
	map_object["settings"]["number_of_replicas"] = replicas
	map_object["mappings"] = {}
	map_object["mappings"][es_type_name] = {}
	my_map = map_object["mappings"][es_type_name]
	my_map["_source"] = {}
	my_map["_source"]["enabled"] = True
	my_map["properties"] = {}
	my_properties = my_map["properties"]

	#Adds a mapping for most non-null fields.
	for column_number in range(column_meta["total_fields"]):
		data_type = column_meta[column_number][DATA_TYPE]
		column_name = column_meta[column_number][NAME]
		column_type = data_types[data_type][DATA_TYPE_NAME]
		column_index = column_meta[column_number][INDEX]
		if column_type in ["null"]:
			pass
		elif column_name in ["LATITUDE", "LONGITUDE"]:
			pass
		else:
			my_properties[column_name] = {}
			my_properties[column_name]["type"] = column_type
			my_properties[column_name]["index"] = column_index
			if column_type == "date":
				my_properties[column_name]["format"] = \
				"YYYYmmdd"
	#Adds a mapping for the geo-point.
	my_properties["pin"] = {}
	my_properties["pin"]["properties"] = {}
	my_properties["pin"]["properties"]["location"] = {}
	my_properties["pin"]["properties"]["location"]["type"] = "geo_point"

	#Add composite fields here.

	return map_object

def get_qs_query(term, fields, boost):
	"""Returns a "query_string" style ElasticSearch query object"""
	qs_query = {}
	qs_query["query_string"] = {}
	qs_query["query_string"]["query"] = term
	qs_query["query_string"]["fields"] = fields
	qs_query["query_string"]["boost"] = boost
	return qs_query
