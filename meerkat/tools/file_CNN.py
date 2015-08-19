import csv

import pandas as pd

from itertools import zip_longest

from meerkat.classification.lua_bridge import get_CNN, load_label_map

BANK_CNN = get_CNN("bank")
CARD_CNN = get_CNN("card")
CARD_SUBTYPE_CNN = get_CNN("card_subtype")
BANK_SUBTYPE_CNN = get_CNN("bank_subtype")

def apply_to_df(reader, classifier, file_name, subtype_classifier):

	first_chunk = True

	for df in reader:

		print("-- Batch --")

		trans = list(df.T.to_dict().values())
		t_len = len(trans)

		if t_len < 128:
			trans = trans + [{"DESCRIPTION":""}] * (128 - t_len)

		trans = classifier(trans, doc_key="DESCRIPTION", label_key="MERCHANT_CNN")

		for t in trans:
			t["DESCRIPTION_alt"] = t["LEDGER_ENTRY"] + " " + t["DESCRIPTION"]

		trans = subtype_classifier(trans, doc_key="DESCRIPTION_alt", label_key="SUBTYPE_CNN")
		trans = trans[0:t_len]

		for t in trans:
			del trans["DESCRIPTION_alt"]

		# Save to file
		out_df = pd.DataFrame(trans)

		if first_chunk:
			out_df.to_csv(file_name, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)
			first_chunk = False
		else:
			out_df.to_csv(file_name, header=False, sep="|", mode="a", encoding="utf-8", index=False, index_label=False)

# Load Dataframe
df = pd.read_csv("data/input/top_50_users.txt", na_filter=False, quoting=csv.QUOTE_NONE, encoding="utf-8", sep="|", error_bad_lines=False)

# Group into bank and card
grouped = df.groupby('CONTAINER', as_index=False)
groups = dict(list(grouped))

# Save groups to separate files and reload as readers
groups["BANK"].to_csv("data/input/bank_top_50_users.csv", sep="|", mode="w", encoding="utf-8", index=False, index_label=False)
groups["CARD"].to_csv("data/input/card_top_50_users.csv", sep="|", mode="w", encoding="utf-8", index=False, index_label=False)
bank_reader = pd.read_csv("data/input/bank_top_50_users.csv", na_filter=False, chunksize=128, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)
card_reader = pd.read_csv("data/input/card_top_50_users.csv", na_filter=False, chunksize=128, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|', error_bad_lines=False)

# Apply Classifiers
apply_to_df(bank_reader, BANK_CNN, "data/output/bank_top_50_users_processed.txt", BANK_SUBTYPE_CNN)
apply_to_df(card_reader, CARD_CNN, "data/output/card_top_50_users_processed.txt", CARD_SUBTYPE_CNN)


			
