#!/usr/local/bin/python3
"""
This moduele takes a .prof file and returns a csv file that contain stats of function
in meerkat (excluding python built-in fucntions)
Created on Jun 13, 2016
@Author: Oscar Pan
"""

########################################## USAGE ####################################

#python3 -m meerkat.profile_code <path_to_.prof_file>
#python3 -m meerkat.profile_code stats.prof

#####################################################################################

import os
import sys
import pstats
import pandas as pd

def get_module_name(stats_list):
	"""Get the .prof file's corresponding module name"""
	for item in stats_list:
		if item[4].startswith("meerkat/"):
			return os.path.basename(item[4]).split(".")[0]
	return "unknow_module"

def run_from_command_line(prof_file_path):
	"""Process .prof file and return module's functions stats in a csv file"""
	header = ["Primitive call", "Total call", "Exclude Subfucntion Time", "Total Time",
		"Module name", "Line Number", "Function Name"]
	profile = pstats.Stats(prof_file_path)
	profile.sort_stats("time")
	# function_stats = {key: profile.stats[key] for key in profile.stats.keys() if "meerkat" in key[0]}
	# beautify the stats
	function_stats = [list(profile.stats[key])[:4] + list(key) for key in profile.stats.keys()
		if "meerkat" in key[0]]
	module_name = get_module_name(function_stats)
	function_stats = pd.DataFrame(function_stats)
	function_stats.columns = header
	function_stats.to_csv(module_name+"_function_stats.csv", index=False)

if __name__ == "__main__":
	run_from_command_line(sys.argv[1])
