{
	"from" : 10,
	"size" : 5,
	"query": {
		"filtered": {
			"filter": {
				"geo_distance": {
					"distance": "12km", 
					"pin.location": {
						"lat": 37.78, 
						"lon": -122.42
					}
				}
			}, 
			"query": {
				"match_all": {}
			}
		}
	}
}
