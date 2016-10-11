"""Fixtures for test_bilstm_tagger"""

def get_config():
	"""Return config"""

	config = {
		"tag_column_map": {
			"MERCHANT_NAME": "merchant",
			"LOCALITY": "city",
			"STORE_NUMBER": "store_number",
			"PHONE_NUMBER": "phone_number",
			"ADDRESS": "address"
		},
		"max_tokens": 35
	}

	return config

def get_token_tag_pairs_input():
	"""Get Input for tokenization and taggin"""

	transactions = [
		({
			"DESCRIPTION": "Debit PIN Purchase ISLAND OF GOLD SUPERMARFRESH MEADOWSNY", 
			"MERCHANT_NAME": "ISLAND OF GOLD SUPERMAR",
			"LOCALITY": "MEADOWS",
			"STATE": "NY"
			},
			["background", "background", "background", "merchant", "merchant", "merchant", "merchant", "city", "state"],
			["Debit", "PIN", "Purchase", "ISLAND", "OF", "GOLD", "SUPERMARFRESH", "MEADOWS", "NY"]),
		({
			"DESCRIPTION": "76", 
			"MERCHANT_NAME": "76"
			},  
			["merchant"], 
			["76"]),
		({
			"DESCRIPTION": "PAYMENT THANK YOU", 
			"MERCHANT_NAME": ""
			}, 
			["background", "background", "background"],
			["PAYMENT", "THANK", "YOU"]),
		({
			"DESCRIPTION": "PAYMENT THANK YOU", 
			"MERCHANT_NAME": "Null"
			}, 
			["background", "background", "background"],
			["PAYMENT", "THANK", "YOU"]),
		({
			"DESCRIPTION": "123 THAI FOOD OAK          HARBOR WA~~08888~~120123052189~~77132~~0~~~0079", 
			"MERCHANT_NAME": "123 THAI FOOD",
			"LOCALITY": "OAK HARBOR",
			"STATE": "WA"
			},
			["merchant", "merchant", "merchant", "city", "city", "state", "background"],
			["123", "THAI", "FOOD", "OAK", "HARBOR", "WA", "~~08888~~120123052189~~77132~~0~~~0079"]),
		({
			"DESCRIPTION": "COX CABLE        ONLINE PMT ***********6POS", 
			"MERCHANT_NAME": "COX CABLE"
			},
			["merchant", "merchant", "background", "background", "background", "background"],
			["COX", "CABLE", "ONLINE", "PMT", "***********6", "POS"]),
		({
			"DESCRIPTION": "AMERICAN EXPRESS DES:SETTLEMENT ID:5049791080                INDN:SUBWAY #29955049791080  CO ID:1134992250 CCD", 
			"MERCHANT_NAME": "AMERICAN EXPRESS, SUBWAY",
			"STORE_NUMBER": "#29955049791080"},
			["merchant", "merchant", "background", "background", "background", "background", "background", "merchant", "store_number", "background", "background" ,"background", "background"],
			["AMERICAN", "EXPRESS", "DES:", "SETTLEMENT", "ID:", "5049791080", "INDN:", "SUBWAY", "#29955049791080", "CO", "ID:" ,"1134992250", "CCD"]),
		({
			"DESCRIPTION": "AA MILES BY POINTS     POINTS.COM    IL", 
			"MERCHANT_NAME": "AA, Points.com",
			"STATE": "IL"
			},
			["merchant", "background", "background", "background", "merchant", "state"],
			["AA", "MILES", "BY", "POINTS", "POINTS.COM", "IL"])
	]

	return transactions

