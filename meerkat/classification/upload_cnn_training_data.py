#!/usr/local/bin/python3.3

"""Bundle a directory of files, that are limited to CSV and JSON files into input.tar.gz and then upload it to an S3 bucket according to our rules

created on April 4, 2016
@author: Feifei Zhu
"""
import datetime
import os
import sys
from plumbum import local

# check files
csv_num, json_exit = 0, False
for filename in os.listdir(sys.argv[1]):
	if filename.endswith('.csv'):
		csv_num += 1
	elif filename.endswith('.json'):
		if json_exit:
			print("should only have one json file")
			sys.exit()
		json_exit = True
	else:
		print("file %s is not csv or json file" %filename)
		sys.exit()
if csv_num == 0:
	print("should at least one csv file")
	sys.exit()
if not json_exit:
	print("should have one json file")
	sys.exit()
print("file check pass")

# tar gz the files
os.chdir(sys.argv[1])
local['tar']['-cvzf']['../input.tar.gz']['.']()
os.chdir('../')
print("files tar.gzed")

# upload the tar.gz file to s3
default_dir_paths = {
		'merchant_card' : "meerkat/cnn/data/merchant/card/",
		'merchant_bank' : "meerkat/cnn/data/merchant/bank/",
		'subtype_card_debit' : "meerkat/cnn/data/subtype/card/debit/",
		'subtype_card_credit' : "meerkat/cnn/data/subtype/card/credit/",
		'subtype_bank_debit' : "meerkat/cnn/data/subtype/bank/debit",
		'subtype_bank_credit' : "meerkat/cnn/data/subtype/bank/credit/",
		'category_bank_debit': "meerkat/cnn/data/category/bank/debit/",
		'category_bank_credit': "meerkat/cnn/data/category/bank/credit/",
		'category_card_debit': "meerkat/cnn/data/category/card/debit/",
		'category_card_credit': "meerkat/cnn/data/category/card/credit/",
	}
dtime = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
dir_path = 's3://s3yodlee/' + default_dir_paths[sys.argv[2]] + dtime + '/'
local['aws']['s3']['cp']['input.tar.gz'][dir_path]()
print("uploaded to s3")

#remove the tar.gz file in local
local['rm']['-f']['input.tar.gz']()
