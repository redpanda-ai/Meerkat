import collections
import csv
from random import choice

file_path = "8000_card_member_matt.csv"
ML_file = open(file_path, encoding="utf-8", errors='replace')
rows = list(csv.DictReader(ML_file, delimiter="|"))
buckets = collections.defaultdict(list)
subsample = []

for row in rows:
	user = row['MEM_ID']
	buckets[user].append(row)

while len(subsample) <= 8000:
	rand_user = choice(list(buckets.keys()))
	subsample.extend(buckets[rand_user])
	print(len(buckets[rand_user]))
	
delimiter = "|"
output_file = open("matt_8000", 'w')
dict_w = csv.DictWriter(output_file, delimiter=delimiter, fieldnames=subsample[0].keys())
dict_w.writeheader()
dict_w.writerows(subsample)
output_file.close()