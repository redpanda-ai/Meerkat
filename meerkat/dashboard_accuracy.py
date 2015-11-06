import json
import random
from meerkat.accuracy import CNN_accuracy, print_results
from meerkat.classification.lua_bridge import get_cnn
from meerkat.various_tools import load_params, load_dict_list, write_dict_list

BANK_MERCHANT_CNN = get_cnn("bank_merchant")
CARD_MERCHANT_CNN = get_cnn("card_merchant")
CARD_DEBIT_SUBTYPE_CNN = get_cnn("card_debit_subtype")
CARD_CREDIT_SUBTYPE_CNN = get_cnn("card_credit_subtype")
BANK_DEBIT_SUBTYPE_CNN = get_cnn("bank_debit_subtype")
BANK_CREDIT_SUBTYPE_CNN = get_cnn("bank_credit_subtype")

def run_from_command_line():
    bank_merchant = "1k_labeled_bank_merchant_samples"
    bank_label_map = "meerkat/classification/label_maps/permanent_bank_label_map.json"
    bank_cnn_map = "meerkat/classification/label_maps/reverse_bank_label_map.json"
    bank_merchant_results = CNN_accuracy(bank_merchant, BANK_MERCHANT_CNN, bank_cnn_map, bank_label_map)
    print_results(bank_merchant_results)

    card_merchant = "1k_labeled_card_merchant_samples"
    card_label_map = "meerkat/classification/label_maps/permanent_card_label_map.json"
    card_cnn_map = "meerkat/classification/label_maps/reverse_card_label_map.json"
    card_merchant_results = CNN_accuracy(card_merchant, CARD_MERCHANT_CNN, card_cnn_map, card_label_map)
    print_results(card_merchant_results)

    bank_debit_subtype = "1k_labeled_bank_debit_samples"
    bank_debit_map = load_params("meerkat/classification/label_maps/bank_debit_subtype_label_map.json")
    bank_debit_reverse_map = __invert_subtype_map(bank_debit_map)
    bank_debit_results = CNN_accuracy(bank_debit_subtype, BANK_DEBIT_SUBTYPE_CNN, bank_debit_map, bank_debit_reverse_map, label_key="Proposed Subtype")
    print_results(bank_debit_results)

    bank_credit_subtype = "1k_labeled_bank_credit_samples"
    bank_credit_map = load_params("meerkat/classification/label_maps/bank_credit_subtype_label_map.json")
    bank_credit_reverse_map = __invert_subtype_map(bank_credit_map)
    bank_credit_results = CNN_accuracy(bank_credit_subtype, BANK_CREDIT_SUBTYPE_CNN, bank_credit_map, bank_credit_reverse_map, label_key="Proposed Subtype")
    print_results(bank_credit_results)

    card_debit_subtype = "1k_labeled_card_debit_samples"
    card_debit_map = load_params("meerkat/classification/label_maps/card_debit_subtype_label_map.json")
    card_debit_reverse_map = __invert_subtype_map(card_debit_map)
    card_debit_results = CNN_accuracy(card_debit_subtype, CARD_DEBIT_SUBTYPE_CNN, card_debit_map, card_debit_reverse_map, label_key="Proposed Subtype")
    print_results(card_debit_results)

    card_credit_subtype = "1k_labeled_card_credit_samples"
    card_credit_map = load_params("meerkat/classification/label_maps/card_credit_subtype_label_map.json")
    card_credit_reverse_map = __invert_subtype_map(card_credit_map)
    card_credit_results = CNN_accuracy(card_credit_subtype, CARD_CREDIT_SUBTYPE_CNN, card_credit_map, card_credit_reverse_map, label_key="Proposed Subtype")
    print_results(card_credit_results)

def __invert_subtype_map(subtype_map):
    reverse_map = {}
    for k, v in subtype_map.items():
        txn_type, txn_sub_type = v.split(" - ")
        reverse_map[txn_sub_type.lower()] = k
    return reverse_map

if __name__ == "__main__":
    run_from_command_line()
