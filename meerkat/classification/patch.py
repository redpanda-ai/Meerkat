import boto
import re

def find_s3_objects_recursively(conn, bucket, my_results, prefix=None, target=None):
	"""Find all S3 target objects and their locations recursively"""
	folders = bucket.list(prefix=prefix, delimiter="/")
	for s3_object in folders:
		if s3_object.name != prefix:
			if s3_object.name[-len(target):] == target:
				my_results[prefix] = target
				return s3_object.name
			elif s3_object.name[-1:] == "/":
				find_s3_objects_recursively(conn, bucket, my_results, prefix=s3_object.name,
					target=target)

def get_peer_models(candidate_dictionary, prefix=None):
	"""Show the candidate models for each peer"""
	results = {}
	my_pattern = re.compile("(" + prefix + ")(.*/)(\d{14}/)")
	for key in my_results:
		#print(key)
		if my_pattern.search(key):
			matches = my_pattern.match(key)
			model_type, timestamp = matches.group(2), matches.group(3)
			if model_type not in results:
				results[model_type] = []
			results[model_type].append(timestamp)
			#print("Found {0}".format(matches.group(2)))
		else:
			print("Not Found")
	return results

if __name__ == "__main__":
	conn = boto.s3.connect_to_region('us-west-2')
	bucket = conn.get_bucket("s3yodlee")

	my_results, prefix = {}, "meerkat/cnn/data"
	find_s3_objects_recursively(conn, bucket, my_results, prefix=prefix, target="results.tar.gz")
	results = get_peer_models(my_results, prefix=prefix)
	for key in sorted(results.keys()):
		print("{0}: {1}".format(key, results[key]))

