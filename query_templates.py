#!/usr/bin/python

#This template is used to count the number of hits in a "query_string" query
count_query = """
{
	"size" : 0,
	"query" : {
		"query_string" : {
			"query" : "__term"
		}
	}
}"""

#This template is used to return select fields from the highest scoring hits from a "query string" query
show_query = """
{
	"from" : 0,
	"size" : 10,
	"fields" : [ 
		"BUSINESSSTANDARDNAME",
		"HOUSE",
		"STREET",
		"STRTYPE",
		"CITYNAME",
		"STATE",
		"ZIP",
		"pin.location" ],
	"query" : {
		"query_string" : {
			"query" : "__term"
		}
	}
}"""


