def get_transaction():
    return [{
        "date": "2014-08-10T00:00:00",
        "description": "taco bell scarsdale, ny",
        "amount": 59.0,
        "transaction_id": 5024853,
        "ledger_entry": "debit"
    }]


def get_transaction_with_subtype():
    return [{
        "date": "2014-08-10T00:00:00",
        "description": "taco bell scarsdale, ny",
        "amount": 59.0,
        "transaction_id": 5024853,
        "ledger_entry": "debit",
        "subtype_CNN": "Payment - Payment"
    }]


def get_transaction_bank_fallback_classifiable():
    return [{
        "CNN": "Con Edison"
    }]


def get_transaction_card_fallback_classifiable():
    return [{
        "CNN": "Legal Sea Foods"
    }]


def get_transaction_subtype_fallback():
    return [{
        "CNN": "Capital One",
        "txn_sub_type": "SSA"
    }]


def get_transaction_subtype_no_fallback():
    return [{
        "CNN": "Capital One",
        "txn_sub_type": "Joseph Rules"
    }]


def fake_bank_subtype_CNN_subtype_payment(transactions):
    return
