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
        "CNN": "Con Edison"
    }]


def get_transaction_card_fallback_classifiable():
    """Return an array containing a single transaction for a merchant whose name only appears in the card fallback map"""
    return [{
        "CNN": "Legal Sea Foods"
    }]


def get_transaction_subtype_fallback():
    """Return an array containing a single transaction for a merchant whose category must be looked up using the transaction subtype"""
    return [{
        "CNN": "Capital One",
        "txn_sub_type": "SSA"
    }]


def get_transaction_subtype_no_fallback():
    """Return an array containing a single transaction for a merchant whose category must be looked up using the transaction subtype and whose transaction subtype does not appearn in the fallback map"""
    return [{
        "CNN": "Capital One",
        "txn_sub_type": "Joseph Rules"
    }]


def get_transaction_subtype_no_merchant():
    """Return an array containing a single transaction with a subtype but no merchant name"""
    return [{
        "txn_sub_type": "ATM Withdrawal"
    }]


def get_transaction_subtype_no_merchant_no_fallback():
    """Return an array containing a single transaction with a subtype but no merchant name and no fallback in the static maps"""
    return [{
        "txn_sub_type": "Joseph Rules"
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
            "description": "some physical location"
        },
        {
            "ledger_entry": "debit",
            "description": "some non-physical location"
        },
        {
            "ledger_entry": "credit",
            "description": "some physical location"
        },
        {
            "ledger_entry": "debit",
            "description": "some non-physical location"
        }
    ]


def get_mock_cnn(transactions, label_key="CNN"):
    """Return a mock CNN which appends a parsable merchant name"""
    for trans in transactions:
        trans[label_key] = "joseph - rules"
    return transactions


def get_mock_sws(description):
    """Return a mock SWS classifier which returns true or false depending on the test data"""
    return description == "some physical location"
