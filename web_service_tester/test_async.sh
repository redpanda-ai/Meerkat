#!/bin/bash

# Usage: Enter a single parameter for how many times you want each post to loop
# Example ./test_async.sh 100

sh ./multi_post.sh big.json v1.0.0 $1 &
sh ./multi_post.sh one_ledger.json v1.0.1 $1 &

