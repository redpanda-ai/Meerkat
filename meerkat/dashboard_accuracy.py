import json
import os
import sys
from meerkat.accuracy import CNN_accuracy, print_results, nostdout, get_s3_connection
from meerkat.classification.lua_bridge import get_cnn
from meerkat.various_tools import load_params
from boto.s3.connection import Location
from elasticsearch import Elasticsearch
from meerkat.various_tools import get_es_connection
import pprint
from datetime import datetime
import logging

BANK_MERCHANT_CNN = get_cnn("bank_merchant")
CARD_MERCHANT_CNN = get_cnn("card_merchant")
CARD_DEBIT_SUBTYPE_CNN = get_cnn("card_debit_subtype")
CARD_CREDIT_SUBTYPE_CNN = get_cnn("card_credit_subtype")
BANK_DEBIT_SUBTYPE_CNN = get_cnn("bank_debit_subtype")
BANK_CREDIT_SUBTYPE_CNN = get_cnn("bank_credit_subtype")

FOLDER = "development/dashboard/"

def initialize():
    """Validates the command line arguments."""
    input_file, params = None, None

    if len(sys.argv) != 2:
        raise InvalidArguments(msg="Supply a single argument for the json"\
        + "formatted configuration file.", expr=None)

    try:
        params = load_params(sys.argv[1])
    except IOError:
        logging.error(sys.argv[1] + " not found, aborting.")
        sys.exit()

    return params

def guarantee_index_and_doc_type(params):
    """Ensure that the index and document type mapping are as they should be"""
    cluster_nodes = params["elasticsearch"]["cluster_nodes"]
    es_index = params["elasticsearch"]["index"]
    es_doc_type = params["elasticsearch"]["type"]

    es_connection = Elasticsearch(cluster_nodes, index=es_index,\
        sniff_on_start=True, sniff_on_connection_fail=True,\
        sniffer_timeout=15, sniff_timeout=15)
    
    if es_connection.indices.exists(index=es_index):
        logging.critical("Index exists, continuing")
        if es_connection.indices.exists_type(index=es_index, doc_type=es_doc_type):
            logging.critical("Doc type exists, continuing")
    else:
        logging.warning("Index does not exist, creating")
        # index_body = params["elasticsearch"]["type_mapping"]
        es_connection.indices.create(index=es_index)
    return es_connection

def run_from_command_line():
    # params = load_params(sys.argv[1])
    # print(len(sys.argv)) # 2
    results_params = initialize()
    # es_connection = get_es_connection(params)
    es_connection = guarantee_index_and_doc_type(results_params)
    pp = pprint.PrettyPrinter(indent = 4)
    es_index = results_params["elasticsearch"]["index"]
    es_doc_type = results_params["elasticsearch"]["type"]

    keys = __get_data_from_s3()

    bank_merchant = "1k_labeled_bank_merchant_samples"
    bank_label_map = "meerkat/classification/label_maps/permanent_bank_label_map.json"
    bank_cnn_map = "meerkat/classification/label_maps/reverse_bank_label_map.json"
    bank_merchant_results = CNN_accuracy(bank_merchant, BANK_MERCHANT_CNN, bank_cnn_map, bank_label_map)
    print_results(bank_merchant_results)

    bank_merchant_body = {}
    bank_merchant_body["results_content"] = bank_merchant_results
    bank_merchant_body["results_type"] = "bank_merchant_results"
    bank_merchant_body["results_date"] = datetime.now()
    # pp.pprint(bank_merchant_body)

    es_connection.index(index=es_index, doc_type=es_doc_type, body=bank_merchant_body)
    # pp.pprint(es_connection.search(index="results_index", body={"query": {"match": {'results_type':'bank_merchant_results'}}}))

    card_merchant = "1k_labeled_card_merchant_samples"
    card_label_map = "meerkat/classification/label_maps/permanent_card_label_map.json"
    card_cnn_map = "meerkat/classification/label_maps/reverse_card_label_map.json"
    card_merchant_results = CNN_accuracy(card_merchant, CARD_MERCHANT_CNN, card_cnn_map, card_label_map)
    print_results(card_merchant_results)

    card_merchant_body = {}
    card_merchant_body["results_content"] = card_merchant_results
    card_merchant_body["results_type"] = "card_merchant_results"
    card_merchant_body["results_date"] = datetime.now()
    # pp.pprint(card_merchant_body)

    es_connection.index(index=es_index, doc_type=es_doc_type, body=card_merchant_body)
    # pp.pprint(es_connection.search(index="results_index", body={"query": {"match": {'results_type':'card_merchant_results'}}}))


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

    __remove_local_data(keys)

def __invert_subtype_map(subtype_map):
    reverse_map = {}
    for k, v in subtype_map.items():
        txn_type, txn_sub_type = v.split(" - ")
        reverse_map[txn_sub_type.lower()] = k
    return reverse_map

def __get_data_from_s3():
    conn = get_s3_connection()
    bucket = conn.get_bucket("s3yodlee", Location.USWest2)
    keys = bucket.list(FOLDER)
    for key in keys:
        if(key.name.lower() == FOLDER.lower()):
            continue
        print("getting data file " + str(key.name))
        filename = key.name.split("/")[-1]
        key.get_contents_to_filename(filename)
        # bucket.get_key(key)
    return keys

def __remove_local_data(keys):
    for key in keys:
        if(key.name.lower() == FOLDER.lower()):
            continue
        filename = key.name.split("/")[-1]
        os.remove(filename)

if __name__ == "__main__":
    run_from_command_line()
    sys.exit(0)
