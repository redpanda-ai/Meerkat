#!/bin/bash

sudo ps -ef | grep python | grep root | grep sudo | grep meerkat.web_service | awk ' { print "sudo kill " $2 } ' | source /dev/stdin

