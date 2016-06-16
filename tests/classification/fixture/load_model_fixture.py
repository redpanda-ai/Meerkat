"""Fixture for test_load_model module"""
from meerkat.various_tools import load_params

def get_trans():
	"""Return trans that contain single transaction"""
	return [{
		"DESCRIPTION_UNMASKED": "USAA FUNDS TRANSFER DB~~03045~~~~59048~~0~~~~0302",
		"LEDGER_ENTRY" : "debit",
		"PROPOSED_SUBTYPE" : "Transfer -Transfer"
	}]


def get_class_size():
	"""Return number of labels for subtype bank debit"""
	return len(load_params("meerkat/classification/label_maps/subtype.bank.debit.json"))

