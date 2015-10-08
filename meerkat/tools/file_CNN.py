import csv
import sys
import math

import pandas as pd
import numpy as np

from meerkat.classification.lua_bridge import get_CNN, load_label_map

#################### USAGE ##########################

# python3 -m meerkat.tools.file_CNN [file_name] [container_type]
# python3 -m meerkat.tools.file_CNN data/input/top_50_users.txt card

#####################################################

BANK_MERCHANT_CNN = get_cnn("bank_merchant")
CARD_MERCHANT_CNN = get_cnn("card_merchant")
CARD_DEBIT_SUBTYPE_CNN = get_cnn("card_debit_subtype")
CARD_CREDIT_SUBTYPE_CNN = get_cnn("card_credit_subtype")
BANK_DEBIT_SUBTYPE_CNN = get_cnn("bank_debit_subtype")
BANK_CREDIT_SUBTYPE_CNN = get_cnn("bank_credit_subtype")

def apply_classifiers(df, merchant_classifier, subtype_classifier):
	"""Apply classifiers to an individual dataframe"""

	print("-- Batch --")

	trans = list(df.T.to_dict().values())
	t_len = len(trans)

	if t_len < 128:
		trans = trans + [{"DESCRIPTION_UNMASKED":""}] * (128 - t_len)

	trans = merchant_classifier(trans, doc_key="DESCRIPTION_UNMASKED", label_key="MERCHANT_CNN")
	trans = subtype_classifier(trans, doc_key="DESCRIPTION_UNMASKED", label_key="SUBTYPE_CNN")
	trans = trans[0:t_len]

	for t in trans:
		t["SUBTYPE_CNN"] = t["SUBTYPE_CNN"].split(" - ")[1]

	return pd.DataFrame(trans)

def apply_to_df(dataframe, file_out):
	"""Shape Dataframe into chunks and apply and save"""

	first_chunk = True

	if sys.argv[2] == "card":
		merchant_classifier = CARD_MERCHANT_CNN
		credit_subtype_classifer = CARD_CREDIT_SUBTYPE_CNN
		debit_subtype_classifer = CARD_DEBIT_SUBTYPE_CNN
	elif sys.argv[2] == "bank":
		merchant_classifier = BANK_MERCHANT_CNN
		credit_subtype_classifer = BANK_CREDIT_SUBTYPE_CNN
		debit_subtype_classifer = BANK_DEBIT_SUBTYPE_CNN
	else:
		print("Please select either bank or card as container")
		sys.exit()

	# Group into credit and debit
	grouped = dataframe.groupby('LEDGER_ENTRY', as_index=False)
	groups = dict(list(grouped))
	credit_split_chunks = np.array_split(groups["credit"], math.ceil(groups["credit"].shape[0] / 128))
	debit_split_chunks = np.array_split(groups["debit"], math.ceil(groups["debit"].shape[0] / 128))

	for df in credit_split_chunks:

		out_df = apply_classifiers(df, merchant_classifier, credit_subtype_classifer)

		if first_chunk:
			out_df.to_csv(file_out, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
			first_chunk = False
		else:
			out_df.to_csv(file_out, header=False, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)

	for df in debit_split_chunks:

		out_df = apply_classifiers(df, merchant_classifier, debit_subtype_classifer)
		out_df.to_csv(file_out, header=False, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)

# Load Dataframe
df = pd.read_csv(sys.argv[1], na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep="|", error_bad_lines=False)
apply_to_df(df, "data/output/CNN_processed.txt")
