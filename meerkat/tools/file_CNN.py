import csv
import pandas as pd

from meerkat.classification.lua_bridge import get_cnn

BANK_CNN = get_cnn("bank")
CARD_CNN = get_cnn("card")
CARD_SUBTYPE_CNN = get_cnn("card_subtype")
BANK_SUBTYPE_CNN = get_cnn("bank_subtype")

def apply_to_df(reader, classifier, file_name, subtype_classifier):

	first_chunk = True

	for each_dataframe in reader:
		print("-- Batch --")
		transactions = list(each_dataframe.T.to_dict().values())
		t_len = len(transactions)

		if t_len < 128:
			transactions = transactions + [{"DESCRIPTION":""}] * (128 - t_len)

		transactions = classifier(transactions, doc_key="DESCRIPTION",\
			label_key="MERCHANT_CNN")

		for each_transaction in transactions:
			each_transaction["DESCRIPTION_alt"] = \
				each_transaction["LEDGER_ENTRY"] + " " + \
				each_transaction["DESCRIPTION"]

		transactions = subtype_classifier(transactions, doc_key="DESCRIPTION_alt",\
		label_key="SUBTYPE_CNN")
		transactions = transactions[0:t_len]

		for each_transaction in transactions:
			del transactions["DESCRIPTION_alt"]

		# Save to file
		out_df = pd.DataFrame(transactions)

		if first_chunk:
			out_df.to_csv(file_name, sep="|", mode="a", \
			encoding="utf-8", index=False, index_label=False)
			first_chunk = False
		else:
			out_df.to_csv(file_name, header=False, sep="|", mode="a", \
			encoding="utf-8", index=False, index_label=False)

# Load Dataframe
MY_DATAFRAME = pd.read_csv("data/input/top_50_users.txt", na_filter=False, \
	quoting=csv.QUOTE_NONE, encoding="utf-8", sep="|", error_bad_lines=False)

# Group into bank and card
GROUPED = MY_DATAFRAME.groupby('CONTAINER', as_index=False)
GROUPS = dict(list(GROUPED))

# Save GROUPS to separate files and reload as readers
GROUPS["BANK"].to_csv("data/input/bank_top_50_users.csv", sep="|", mode="w",\
	encoding="utf-8", index=False, index_label=False)
GROUPS["CARD"].to_csv("data/input/card_top_50_users.csv", sep="|", mode="w",\
	encoding="utf-8", index=False, index_label=False)
BANK_READER = pd.read_csv("data/input/bank_top_50_users.csv", na_filter=False,\
	chunksize=128, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|',\
	error_bad_lines=False)
CARD_READER = pd.read_csv("data/input/card_top_50_users.csv", na_filter=False,\
	chunksize=128, quoting=csv.QUOTE_NONE, encoding="utf-8", sep='|',\
	error_bad_lines=False)

# Apply Classifiers
apply_to_df(BANK_READER, BANK_CNN,\
	 "data/output/bank_top_50_users_processed.txt", BANK_SUBTYPE_CNN)
apply_to_df(CARD_READER, CARD_CNN,\
	 "data/output/card_top_50_users_processed.txt", CARD_SUBTYPE_CNN)

