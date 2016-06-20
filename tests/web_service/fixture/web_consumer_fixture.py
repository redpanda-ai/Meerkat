import uuid
import json

def get_transactions_to_clean(case_type):
    """Return an array of transactions to be cleaned fo proper schema"""
    if case_type == "non_physical_no_debug":
        return [{
            "is_physical_merchant": False,
            "transaction_id": 123,
            "txn_sub_type": "Purchase",
            "txn_type": "Purchase",
            "locale_bloom": None
        }]
    elif case_type == "physical_no_debug":
        return [{
            "is_physical_merchant": True,
            "transaction_id": 123,
            "txn_sub_type": "Purchase",
            "txn_type": "Purchase",
            "CNN": "Ikea",
            "country": "US",
            "locale_bloom": None
        }]
    elif case_type == "physical_debug":
        return [{
            "is_physical_merchant": True,
            "transaction_id": 123,
            "txn_sub_type": "Purchase",
            "txn_type": "Purchase",
            "CNN": "Ikea",
            "country": "US",
            "state": "NY",
            "city": "Scars",
            "search": {},
            "locale_bloom": ["Scars", "NY"]
        }]

def get_proper_output(case_type):
    """Return an array of proper output"""
    base_dir = "tests/web_service/fixture/"
    file_path = base_dir + case_type + ".json"
    json_file = open(file_path, encoding='utf-8')
    return [json.load(json_file)]

def get_transaction():
    """Create and return an array containing a single transaction"""
    return [{
        "date": "2014-08-10T00:00:00",
        "description": "taco bell scarsdale, ny",
        "amount": 59.0,
        "transaction_id": 5024853,
        "ledger_entry": "debit"
    }]

def get_transaction_with_subtype():
    """Create and return an array containing a single transaction with a subtype appended"""
    return [{
        "date": "2014-08-10T00:00:00",
        "description": "taco bell scarsdale, ny",
        "amount": 59.0,
        "transaction_id": 5024853,
        "ledger_entry": "debit",
        "subtype_CNN": "Payment - Payment"
    }]

def get_transaction_bank_fallback_classifiable():
    """Return an array containing a single transaction for a merchant whose name only appears in the bank fallback map"""
    return [{
        "CNN": {"label": "Con Edison", "category": "Utilities"}
    }]

def get_transaction_card_fallback_classifiable():
    """Return an array containing a single transaction for a merchant whose name only appears in the card fallback map"""
    return [{
        "CNN": {"label": "Legal Sea Foods", "category": "Restaurants/Dining"}
    }]

def get_transaction_subtype_fallback():
    """Return an array containing a single transaction for a merchant whose category must be looked up using the transaction subtype"""
    return [{
        "CNN": {"label": "Capital One", "category": "Use Subtype Rules for Categories"},
        "subtype_CNN": {"label": "Deposits & Credits - SSA", "category": "Other Income"}
    }]

def get_transaction_subtype_no_fallback():
    """Return an array containing a single transaction for a merchant whose category must be looked up using the transaction subtype and whose transaction subtype does not appear in the fallback map"""
    return [{
        "CNN": {"label": "Capital One", "category": "Use Subtype Rules for Categories"},
        "subtype_CNN": {"label": "Joseph Rules", "category": "Joseph Rules"}
    }]

def get_transaction_subtype_no_merchant():
    """Return an array containing a single transaction with a subtype but no merchant name"""
    return [{
        "subtype_CNN": {"label": "Withdrawal - ATM Withdrawal", "category": "Cash Withdrawal"}
    }]

def get_transaction_subtype_no_merchant_no_fallback():
    """Return an array containing a single transaction with a subtype but no merchant name and no fallback in the static maps"""
    return [{
        "subtype_CNN": {"label": "Joseph Rules"}
    }]

def get_transaction_subtype_non_existant_merchant():
    """Return an array containing a single transaction with a merchant that doesn't exist in the subtype map"""
    return [{
        "CNN": {"label": "Joseph Rules"}
    }]

def get_test_request_bank():
    """return an API request with the "bank" container"""
    return {
        "container": "bank",
        "transaction_list": get_test_transaction_list()
    }

def get_test_request_card():
    """return an API request with the "card" container"""
    return {
        "container": "card",
        "transaction_list": get_test_transaction_list()
    }

def get_test_transaction_list():
    """Return a list of transactions which will test different paths in the web_consumer"""
    return [
        {
            "ledger_entry": "credit",
            "description": "some physical location",
            "locale_bloom": ["Mockville", "CA"]
        },
        {
            "ledger_entry": "debit",
            "description": "some physical location",
            "locale_bloom": ["Mockville", "CA"]
        },
        {
            "ledger_entry": "credit",
            "description": "some non-physical location",
            "locale_bloom": ["Mockville", "CA"]
        },
        {
            "ledger_entry": "debit",
            "description": "some non-physical location",
            "locale_bloom": ["Mockville", "CA"]
        }
    ]

def get_mock_hyperparams():
    """Return a mock object with the necessary fields of the web_consumer's hyperparams.  Consider loading regular hyperparams from json"""
    return {
        "es_result_size": "45",
        "z_score_threshold": "2.857",
        "raw_score_threshold": "1.000",
        "good_description": "2",
        "boost_labels": ["standard_fields"],
        "boost_vectors": {
            "address": [0.541],
            "address_extended": [1.282],
            "admin_region": [0.69],
            "category_labels": [1.319],
            "chain_name": [0.999],
            "email": [0.516],
            "internal_store_number": [1.9],
            "locality": [1.367],
            "name": [2.781],
            "neighborhood": [0.801],
            "po_box": [1.292],
            "post_town": [0.577],
            "postcode": [0.914],
            "region": [1.685],
            "tel": [0.597]
        }
    }

def get_mock_params():
    """Return an object with the necessary fields of the web_consumer's params.  Consider loading the regular params from json"""
    return {
        "input": {
            "hyperparameters": "test",
            "encoding": "utf-8"
        },
        "output": {
            "results": {
                "fields": [
                    "name", "category_labels", "address", "locality", "country",
                    "region", "postcode", "factual_id", "internal_store_number",
                    "latitude", "longitude"],
                "labels": [
                    "merchant_name", "category_labels", "street", "city", "country",
                    "state", "postal_code", "source_merchant_id", "store_id",
                    "latitude", "longitude"]
            }
        },
        "elasticsearch": {
            "cluster_nodes": [
                "test"
            ],
            "index": "factual_index",
            "type": "factual_type"
        },
        "gpu_mem_fraction": 0.33
    }

def get_mock_esconnection(params):
    index = params["elasticsearch"]["index"]
    index_type = params["elasticsearch"]["type"]
    mapping = {index: {"mappings": {index_type: {}}}}
    indices = lambda: None
    indices.get_mapping = lambda: mapping
    es = lambda: None
    es.indices = indices
    return es

def get_mock_msearch(queries):
    """Return a mock literal search method which creates a mock response for every query"""
    queries = queries.split("\n")
    responses = []
    for query in queries:
        if "index" in query:
            continue
        responses.append(get_mock_factual())
    return {
        "responses": responses
    }

def get_mock_factual():
    """Return a mock factual response"""
    hits = [get_mock_hit(10)]
    for i in range(9):
        hits.append(get_mock_hit(1))
    return {
        "took": 31,
        "timed_out": False,
        "hits": {
            "total": 1,
            "max_score": 4.82026,
            "hits": hits
        }
    }

def get_mock_hit(score):
    randomId = uuid.uuid4()
    return {
        "fields": {
            "postcode": [
                "12345"
            ],
            "name": [
                "Mock Merchant"
            ],
            "locality": [
                "Mockville"
            ],
            "region": [
                "CA"
            ],
            "category_labels": [
                "[\"Retail\",\"Gift and Novelty\"]"
            ],
            "factual_id": [
                randomId.hex
            ],
            "address": [
                "123 Mock St"
            ],
            "country": [
                "us"
            ]
        },
        "_source": {
            "pin": {
                "location": {
                    "coordinates": [
                        "-73.807267",
                        "40.989156"
                    ],
                    "type": "point"
                }
            }
        },
        "_type": "factual_type",
        "_index": "factual_index",
        "_id": randomId.hex,
        "_score": score
    }

def get_mock_sws(description):
    """Return a mock SWS classifier which returns true or false depending on the test data"""
    return description == "some physical location"

def get_mock_cnn(transactions, label_key="CNN", label_only=False):
    """Return a mock CNN which appends a parsable merchant name"""
    for trans in transactions:
        trans[label_key] = {"label": "joseph - rules", "category": "ruling class"}
    return transactions

