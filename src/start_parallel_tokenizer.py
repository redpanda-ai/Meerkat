#!/usr/local/bin/python3

"""This script scans, tokenizes, and constructs queries to match transaction
description strings (unstructured data) to merchant data indexed with
ElasticSearch (structured data)."""

import tokenize_descriptions

tokenize_descriptions.start()
