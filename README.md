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
* Python 3.3
* Pandas
* Numpy
* SciPy
* Scikit-Learn
* Elasticsearch
* Boto

License information
-------------------
All rights reserved.

Contributors
------------
* J. Andrew Key, Matt Sevrens, and M. Phani Ram
