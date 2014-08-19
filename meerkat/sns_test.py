import boto.sns
import json
import sys

REGION = 'us-west-2'
TOPIC = 'arn:aws:sns:us-west-2:003144629351:panel_6m'
MESSAGE = sys.argv[1]
print("Sending SNS notification")
print("Topic: {0}".format(TOPIC))
print("Message: {0}".format(MESSAGE))
conn = boto.sns.connect_to_region(REGION)
pub = conn.publish(topic = TOPIC, message = MESSAGE)
print("Notification Sent")
#Pretty cool, list of all subscribers to topic
#x = conn.get_all_subscriptions_by_topic(TOPIC)
#print(x)
