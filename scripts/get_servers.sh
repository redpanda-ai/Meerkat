#!/bin/bash

ec2-describe-instances | awk ' { print $15 } ' | sed '/^\s*$/d' | sort
