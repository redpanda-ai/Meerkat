Longtail
=====
Python project for labeling and classifying Longtail data

Usage
------------
**Run Longtail Classifier:**<br>
From base directory execute **python3 -m longtail config/{config file}**

**Train Binary Classifier:**<br>
From base directory execute **python3 -m longtail.binary_classifier.train**<br>
Accepts one optional argument, a CSV containing desripitions labeled as physical or non physical<br>
Defaults to data/misc/verifiedLabeledTrans.csv

**Hand Labeled Transactions:**<br>
The Longtail classifier uses a combination of supervised learning algorithms and basic search. 
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
* Python 3
* Numpy
* SciPy
* Scikit-Learn
* Elasticsearch

License information
-------------------
All rights reserved.

Contributers
------------
* Matt Sevrens
* J. Andrew Key
