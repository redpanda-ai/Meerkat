# To generate an egg with no source issue the following commnand:
# python3.3 setup.py bdist_egg --exclude-source-files

from setuptools import setup, find_packages
setup(
	name = 'longtail',
	version = '0.1.c6',
	packages = find_packages(exclude=['tests', 'json_queries']),

	#Other stuff
	install_requires = [
		'elasticsearch>=1.0.0',
		'matplotlib==1.3.1',
		'nose==1.3.1',
		'numpy==1.8.0',
		'scipy==0.13.3',
		'urllib3==1.8',
	],
	# metadata for upload to PyPi
	author = 'me',
	author_email = 'me@example.com',
	)

