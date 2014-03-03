Longtail
=====
Python project for labeling and classifying Longtail data

Usage
------------
From base directory execute python3 -m longtail config/{config file}

Installation
------------
Manually for now

=======
**Run Longtail Classifier:**<br>
From base directory execute **python3 -m longtail config/{config file}**

**Train Binary Classifier:**<br>
From base directory execute **python3 -m longtail.binary_classifier.train**<br>
Accepts one optional argument, a CSV containing desripitions labeled as physical or non physical<br>
Defaults to data/misc/verifiedLabeledTrans.csv

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
