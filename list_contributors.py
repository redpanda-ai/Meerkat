"""This module will create a csv file with a list of current contributors"""

import subprocess
from pandas import DataFrame

def main_program():
	"""Create a csv file with a list of current contributors for meerkat python modules"""
	all_py_files = subprocess.check_output(
		'git ls-tree -r --name-only develop ./meerkat/ | grep "py$"',
		stderr=subprocess.STDOUT,
		shell=True
	)

	current_contributors = {
		'J. Andrew Key': 'J. Andrew Key',
		'J. A. Key': 'J. Andrew Key',
		'Matt Sevrens': 'Matt Sevrens',
		'msevrens': 'Matt Sevrens',
		'Oscar Pan': 'Oscar Pan',
		'opan': 'Oscar Pan',
		'Feifei Zhu': 'Feifei Zhu',
		'ffeizhu': 'Feifei Zhu',
		'Jie Zhang': 'Jie Zhang',
		'Tina Wu': 'Tina Wu',
		'diwu001': 'Tina Wu'
	}

	file_names = all_py_files.decode('utf-8').split('\n')
	file_names = [name for name in file_names if name]
	contributors_list = []
	counts = []

	for file_name in file_names:
		contributor_names = subprocess.check_output(
			'git shortlog -s -- ' + file_name + r' | sed -e "s/^\s*[0-9]*\s*//"',
			stderr=subprocess.STDOUT,
			shell=True
		)
		processed_names = contributor_names.decode('utf-8').split('\n')
		valid_names = set()
		for name in processed_names:
			if name in current_contributors:
				valid_names.add(current_contributors[name])
		contributors_list.append(list(valid_names))
		counts.append(len(valid_names))

	df = DataFrame({'Module Name': file_names, 'Current Contributors': contributors_list,
		"Number of Contributors": counts})
	df = df.sort_values(by="Number of Contributors")
	df.to_csv('contributors.csv', index=False)

if __name__ == "__main__":
	main_program()
