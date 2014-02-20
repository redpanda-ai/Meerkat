echo -e "\nDeleting index and type mapping"
curl -XDELETE brainstorm8:9200/ex_index
echo -e "\nCreating a new type with a geo-shape mapping"
curl -XPUT brainstorm8:9200/ex_index -d '{
	"mappings": {
		"ex_type": {
			"_source": {
				"enabled": true
			},
			"properties": {
				"name": {
					"index": "analyzed",
					"type": "string"
				},
				"pin": {
					"properties": {
						"location": {
							"type": "geo_shape",
							"tree": "quadtree",
							"precision": "1m"
						}
					}
				}
			}
		}
	},
	"settings": {
		"number_of_replicas": 1,
		"number_of_shards": 12
	}
}'
echo -e "\nConfirming the mapping."
curl -XGET brainstorm8:9200/ex_index/_mapping?pretty=true
echo -e "\nAdding a point"
curl -XPUT brainstorm8:9200/ex_index/ex_type/1 -d '{
	"name": "Andy",
	"pin" : {
		"location": {
			"type": "point",
			"coordinates" : [2, 3]
		}
	}
}'
echo -e "\nAdding a second point"
curl -XPUT brainstorm8:9200/ex_index/ex_type/2 -d '{
	"name": "Bobbi",
	"pin" : {
		"location": {
			"type": "point",
			"coordinates" : [5, 6]
		}
	}
}'
sleep 1
echo -e "\nConfirming both points"
curl -XGET brainstorm8:9200/ex_index/ex_type/_search?pretty=true -d '{
	"fields" : ["name", "pin.location"],
	"query" : {
		"match_all" : {}
	}
}'
echo -e "\nSearching within an envelope, should return only 'Andy'"
curl -XGET brainstorm8:9200/ex_index/ex_type/_search?pretty=true -d '{
	"fields" : ["name", "pin.location"],
	"query" : {
		"geo_shape" : {
			"pin.location" : {
				"shape" : {
					"type" : "envelope",
					"coordinates": [[1,4],[3,2]]
				}
			}
		}
	}
}'
echo -e "\nSearching within a polygon, should return only 'Bobbi'"
curl -XGET brainstorm8:9200/ex_index/ex_type/_search?pretty=true -d '{
	"fields" : ["name", "pin.location"],
	"query" : {
		"geo_shape" : {
			"pin.location" : {
				"shape" : {
					"type" : "polygon",
					"coordinates": [
						[ [1,3], [3,7], [10,7], [10,4], [1,3] ]
					]
				}
			}
		}
	}
}'

