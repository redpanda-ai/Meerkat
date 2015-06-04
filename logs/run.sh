#!/bin/bash

# description: this script will cycle the log files in a given logs directory
# the only required architecture is that logs/, cycle.py, run.sh are
# in the same dictory, and there exists a .tmp folder inside the logs filder

while true
	do
		# cycle the files
		python3 cycle.py
		rm logs/*.log
		mv logs/.tmp/* logs/

		# the date being recorded in the file here is arbitrary
		# what's important is that the day7.log file is being recorded
		# at this point in time
		date > logs/day7.log

		FILES=logs/*

		for f in $FILES
		do
			head "$f"
		done

		# wait 24 hours
		echo ""
		sleep 2
	done