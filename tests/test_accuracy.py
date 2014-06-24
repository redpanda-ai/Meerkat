"""Unit tests for meerkat.accuracy"""

import unittest, datetime, pprint, json
from meerkat.accuracy import test_accuracy, speed_tests, print_results

class AccuracyTests(unittest.TestCase):

	"""Our UnitTest class."""

	config = """{
		"concurrency" : 1,
		"verification_source" : "data/misc/matt_8000_card.txt",
		"input" : {
			"filename" : "data/input/100_bank_transaction_DESCRIPTION_UNMASKEDs.txt",
			"encoding" : "utf-8",
			"delimiter" : "|"
		},
		"logging" : {
			"level" : "warning", "path" : "logs/foo.log", "console" : false,
			"formatter" : "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
		},
		"output" : {
			"results" : {
				"fields" : ["name", "factual_id", "pin.location", "locality", "region"],
				"size" : 1
			},
			"file" : {
				"format" : "csv", "path" : "data/output/meerkatLabeled.csv"
			}
		},
		"elasticsearch" : {
			"cluster_nodes" : [
		    "s01:9200",
		    "s02:9200",
		    "s03:9200",
		    "s04:9200",
		    "s05:9200",
		    "s06:9200",
		    "s07:9200",
		    "s08:9200",
		    "s09:9200",
		    "s10:9200",
		    "s11:9200",
		    "s12:9200",
		    "s13:9200",
		    "s14:9200",
		    "s15:9200",
		    "s16:9200",
		    "s17:9200",
		    "s18:9200"
    	],
			"index" : "factual_index", "type" : "factual_type",
			"boost_labels" : [ "standard_fields", "composite.address" ],
			"boost_vectors" : {
				"factual_id" :        [ 0.0, 1.0 ],
				"name" :              [ 1.0, 0.0 ],
				"address" :           [ 0.0, 1.0 ]
			}
		},
		"search_cache" : {}
	}"""

	physical = [{'locality': 'San Francisco', 'name': "Smuggler's Cove", 'DESCRIPTION_UNMASKED': "CHECKCARD 1130 SMUGGLER'S COVE SAN FRANCISCOCA 24431063335207088500107", 'factual_id': 'de28a470-d829-012e-561a-003048cad9da', 'composite.address': '650 Gough St  San Francisco, CA 94102, us'}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': "MEL'S DRIVE-IN #2 SAN FRANCISCOCA 2449", 'factual_id': '', 'composite.address': ''}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'CHECKCARD 0121 GREEN PAPAYA RESTAURANT SAN FRANCISCOCA..', 'factual_id': '', 'composite.address': ''}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'BALOMPIE CAFE RESTA SAN FRANCISCO   CAUS', 'factual_id': '', 'composite.address': ''}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'CHECKCARD 0527 GHIRARDELLI #138- GCE SAN FRANCISCOCA 24493983', 'factual_id': '', 'composite.address': ''}, {'locality': 'San Francisco', 'name': '21st Amendment Brewery Cafe', 'DESCRIPTION_UNMASKED': 'CHECKCARD 0308 21ST AMENDMENT BRE SAN FRANCISCOCA 24224433068105013518906', 'factual_id': '44304903-661d-49f1-adbb-c0a73eaf886c', 'composite.address': '563 2nd St " 2nd" San Francisco, CA 94107, us'}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'CHECKCARD 0813 DR CHARLES MOLOSKY DDS SAN FRANCISCOCA...', 'factual_id': '', 'composite.address': ''}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'HERBIVORE           SAN FRANCISCO   CAUS', 'factual_id': '', 'composite.address': ''}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'THRIFT TOWN #3         SAN FRANCISCOCAUS', 'factual_id': '', 'composite.address': ''}, {'locality': 'San Francisco', 'name': 'Gary Danko', 'DESCRIPTION_UNMASKED': 'CHECKCARD 0804 GARY DANKO SAN FRANCISCOCA 24224433218101045757934', 'factual_id': '8759900f-86b3-4e19-a487-3cc832428936', 'composite.address': '800 N Point St  San Francisco, CA 94109, us'}, {'locality': 'San Francisco', 'name': "John's Grill", 'DESCRIPTION_UNMASKED': "CHECKCARD 0801 JOHN'S GRILL SAN FRANCISCOCA 24055233215206112100206", 'factual_id': '90143837-d525-4c5f-9e76-3cfeeb9adf4f', 'composite.address': '63 Ellis St  San Francisco, CA 94102, us'}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'ATM Withdrawal BENDERS BAR     806 S VAN NESS AVE     SAN FRAN     CAUS', 'factual_id': '', 'composite.address': ''}, {'locality': '', 'name': '', 'DESCRIPTION_UNMASKED': 'CHECKCARD 0827 FOG HARBOR FISH HOUSE SAN FRANCISCOCA 24071053240158141237098', 'factual_id': '', 'composite.address': ''}]
	non_physical = ['"CHECKCARD 0120 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0221 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0321 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0420 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0426 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0520 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0621 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0720 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0820 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 0920 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 1020 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 1120 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 1220 24HOUR FITNESS USA,INC 800-432-6348 CA..."', '"CHECKCARD 1227 BCF-COQUITLAM,QUEEN OF VICTORIA BC 74500012362461680275931"', '"CHECKCARD 1227 BCF-COQUITLAM,QUEEN OF VICTORIA BC 74500012362461680275931..."', '"CONCUR DES:EXPENSE ID:C008N1RUX000 INDN:Key, Joe A. CO ID:1911608052 PPD..."', '"CONCUR DES:EXPENSE ID:C00A9UJCX006 INDN:Key, Joe A. CO ID:1911608052 PPD..."', '"H S B C 04/12 #000887347 WITHDRWL H S B C / JOSE AZUETA, FEE"', '"H S B C 04/12 #000887347 WITHDRWL H S B C / JOSE AZUETA, INTERNATIONAL..."', '"H S B C 04/12 #000887347 WITHDRWL H S B C / JOSE AZUETA,"', '"YODLEE.COM INC DES:DIRECT DEP ID:315018338477QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:475019277469QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:515046671679QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:516042312983QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:516042312984QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:517046621180QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:519046304065QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:569025025433QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:575027196035QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:588025780457QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:590017342045QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:612026847301QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:625043170387QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:636041888967QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:640025839973QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:640025839974QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:654038835037QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:669033410119QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:681023381652QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:689051101128QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:696048239467QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:703030941426QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:714044355791QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:715043619496QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:740017497838QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:740017497839QBC INDN:KEY,JOE A CO..."', '"YODLEE.COM INC DES:DIRECT DEP ID:793028169125QBC INDN:KEY,JOE A CO..."', '*SCOTIABANK IN 04/07 #000684996 WITHDRWL SCOTIABANK IN/ BCO INVERLAT', '*SCOTIABANK IN 04/07 #000685080 WITHDRWL SCOTIABANK IN/ BCO INVERLAT', '*SCOTIABANK IN 04/08 #000704407 WITHDRWL SCOTIABANK IN/ BCO INVERLAT', '7-ELEVEN        77 NEWELL RD           EAST PALO ALTCAUS', 'BKOFAMERICA ATM 04/07 #000008734 WITHDRWL LAX-TERMINAL 5 D LOS ANGELES...', 'BNP 10/24 #000766600 WITHDRWL >49740AGENCE P PARIS 09', 'CHASE 06/27 #000530159 WITHDRWL 2600 CYPRESS ST W MONROE LA']
	params = json.loads(config)
	accuracy_results = test_accuracy(params, non_physical_trans=non_physical, result_list=physical)
	time_delta = datetime.datetime.now() + datetime.timedelta(seconds=-100)
	speed_results = speed_tests(time_delta, accuracy_results)

	def setUp(self):
		self.accuracy_results = test_accuracy(self.params, non_physical_trans=self.non_physical, result_list=self.physical)

	def test_precision(self):
		self.assertEqual(self.accuracy_results['precision'], 100)

	def test_total_processed(self):
		self.assertEqual(self.accuracy_results['total_processed'], 67)

	def test_total_physical(self):
		self.assertEqual(round(self.accuracy_results['total_physical']), 19)

	def test_total_non_physical(self):
		self.assertEqual(round(self.accuracy_results['total_non_physical']), 81)

	def test_binary_accuracy(self):
		self.assertEqual(self.accuracy_results['binary_accuracy'], 100)

	def test_total_recall(self):
		self.assertEqual(round(self.accuracy_results['total_recall']), 6)

	def test_total_recall_physical(self):
		self.assertEqual(round(self.accuracy_results['total_recall_physical']), 31)

	def test_time_taken(self):
		self.assertEqual(self.speed_results['time_delta'].seconds, 100)

	def test_time_per_transaction(self):
		self.assertEqual(round(self.speed_results['time_per_transaction'], 1), 1.5)
	
	def test_transactions_per_minute(self):
		self.assertEqual(round(self.speed_results['transactions_per_minute']), 40)
		
if __name__ == '__main__':
	unittest.main(argv=[sys.argv[0]])
