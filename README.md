Meerkat
=====
Python project for labeling and classifying financial transactions

Usage
------------
**Load an ElasticSearch Index:**<br>
From the base directory execute **python3 -m meerkat.bulk_loader config/{config file}**

**Run Meerkat Classifier:**<br>
From base directory execute **python3 -m meerkat config/{config file}**

**Train Binary Classifier:**<br>
From base directory execute **python3 -m meerkat.binary_classifier.train**<br>
Accepts one optional argument, a CSV containing desripitions labeled as physical or non physical<br>
Defaults to data/misc/verifiedLabeledTrans.csv

**Hand Labeled Transactions:**<br>
The Meerkat classifiers use a combination of supervised learning algorithms and basic search. 
The labeled data requires three columns: DESCRIPTION, factual_id and IS_PHYSICAL_TRANSACTION.

DESCRIPTION is the transaction itself, factual_id is a unique identifier linking to the
factual.com record that the transaction represents, and IS_PHYSICAL_TRANSACTION is a binary
label which represents whether the transaction occurred in a brick and mortar location
or otherwise. 

IS_PHYSICAL_TRANSACTION is used by the binary classfier, while factual_id
is used to generate an accuracy score which then in turn is used to optimize our hyperparameters.
The preferred location for verified transactions is under data/misc/verifiedLabeledTrans.csv 
though a custom file may be passed in as well. 

Installation
------------
Manually for now

Dependencies
------------
#### Linux Version
<pre>
[root@ip-172-31-25-23 ~]# uname -r
2.6.32-431.29.2.el6.x86_64
</pre>
#### Linux Distribution
<pre>
[root@ip-172-31-25-23 Meerkat]# cat /etc/centos-release
CentOS release 6.5 (Final)
</pre>
#### Python version
<pre>
[root@ip-172-31-25-23 ~]# python3.3 --version
Python 3.3.5
</pre>
#### Python Libraries
<pre>
[root@ip-172-31-25-23 Meerkat]# pip3.3 freeze
Menu==1.4
Tornado-JSON==0.41
astroid==1.1.1
awscli==1.5.3
bcdoc==0.12.2
bitarray==0.8.1
boto==2.9.9
botocore==0.67.0
colorama==0.2.5
docutils==0.12
ecdsa==0.11
elasticsearch==1.0.0
jmespath==0.4.1
jsonschema==2.4.0
logilab-common==0.61.0
matplotlib==1.3.1
nltk==3.0.0
nose==1.3.3
numpy==1.8.1
pandas==0.14.1
paramiko==1.15.1
progressbar==2.2
pyasn1==0.1.7
pybloom==2.0.0
pycrypto==2.6.1
pylint==1.2.1
pyparsing==2.0.2
python-dateutil==2.2
pytz==2014.7
rsa==3.1.2
scikit-learn==0.15.2
scipy==0.14.0
six==1.6.1
tornado==3.2.1
urllib3==1.8.2
virtualenv==1.11.6
</pre>
#### Elasticsearch and Lucene versions
<pre>
[root@ip-172-31-25-23 Meerkat]# curl -s 172.31.26.85:9200
{
  "status" : 200,
  "name" : "Frigga",
  "version" : {
    "number" : "1.3.1",
    "build_hash" : "2de6dc5268c32fb49b205233c138d93aaf772015",
    "build_timestamp" : "2014-07-28T14:45:15Z",
    "build_snapshot" : false,
    "lucene_version" : "4.9"
  },
  "tagline" : "You Know, for Search"
}
</pre>

License information
-------------------
All rights reserved.

Contributors
------------
* J. Andrew Key
* Matt Sevrens
* M. Phani Ram
* As Shaja
* Ashish Kulkarni
