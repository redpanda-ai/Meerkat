""" Just a test bed for new ideas."""

from boto.s3.connection import Key, Location
from boto import connect_s3

def pull_from_s3(*args, **kwargs):
	conn = connect_s3()
	bucket = conn.get_bucket(kwargs["bucket"], Location.USWest2)
	listing = bucket.list(prefix=kwargs["prefix"])

	my_filter = kwargs["filter"]
	my_listing = [
		s3_object
		for s3_object in listing
		if s3_object.key[-len(my_filter):] == my_filter
	]
	get_filename = lambda x: x.key[x.key.rfind("/")+1:]
	for s3_object in my_listing:
		get_filename(s3_object)
		print("Local Filename: {0}, S3Key: {1}".format(get_filename(s3_object), s3_object))
		s3_object.get_contents_to_filename(get_filename(s3_object))

""" Main program"""
if __name__ == "__main__":
	pull_from_s3(bucket="yodleemisc", prefix="hvudumala/Type_Subtype_finaldata/Card/", filter="csv")


