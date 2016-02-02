""" Just a test bed for new ideas."""

from boto.s3.connection import Key, Location
from boto import connect_s3

#def pull_from_s3(bucket_name, prefix, extension):
def pull_from_s3(*args, **kwargs):
	conn = connect_s3()
	bucket = conn.get_bucket(kwargs["bucket"], Location.USWest2)
	listing = bucket.list(prefix=kwargs["prefix"])

	my_filter = kwargs["filter"]
	my_listing = [x.key[x.key.rfind("/")+1:] for x in listing if x.key[-len(my_filter):] == my_filter]
	print(my_listing)
#	for item in listing:
#		if item.key[-len(my_filter):] == my_filter:
#			my_index = item.key.rfind("/")
#			print(item.key[my_index+1:]) 

""" Main program"""
if __name__ == "__main__":
	pull_from_s3(bucket="yodleemisc", prefix="hvudumala/Type_Subtype_finaldata/Card/", filter="csv")

