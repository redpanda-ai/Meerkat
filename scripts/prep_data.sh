#!/bin/bash

ssh -i ~/.ssh/meerkat.pem ${1} "mkfs -t ext4 /dev/xvdb"
ssh -i ~/.ssh/meerkat.pem ${1} "mount /dev/xvdb /data"
ssh -i ~/.ssh/meerkat.pem ${1} "df | grep 'data'"
ssh -i ~/.ssh/meerkat.pem ${1} "chown -R elasticsearch /data"


