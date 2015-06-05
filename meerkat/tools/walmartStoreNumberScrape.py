"""
Scrapes data about at most 6,000 Walmart stores. Gets the following pieces of
information:
store Number
address
city
state
zip code

The file is then saved into
/data/misc/Store\ Numbers/Clean/mcdonalds_store_numbers.txt"
"""

from splinter.browser import Browser
from splinter.exceptions import ElementDoesNotExist
from pprint import pprint
# import os.path

def run():
	browser = Browser()
	base_url = "http://www.walmart.com/storeLocator/ca_storefinder_details_short.do?edit_object_id="
	current_walmart = 1755
	found = []
	save_file = open("../../data/misc/Store\ Numbers/Clean/walmart_store_numbers.txt", "a")

	# Clear Contents of Old Run
	#open('walmart_store_numbers.txt', 'w').close()

	while current_walmart < 6000:

		text_number = str(current_walmart)
		while len(text_number) < 4:
			text_number = "0" + text_number

		browser.visit(base_url + str(current_walmart))

		try:

			# Find Address on Page
			elements = browser.find_by_css(".StoreAddress")
			div = elements[0]

			# Parse
			_, address, extra = div.value.split('\n')
			city, extra = extra.split(",")
			state, zip_code = extra.lstrip(' ').split(' ')

			# Format
			new = {
			"internal_store_number" : ("#" + text_number),
			"address" : address,
			"city" : city,
			"state" : state,
			"zip_code" : zip_code
			}

			found.append(new)

			# Save Result
			pprint(new, save_file)

			# Print
			print(str(current_walmart), ":")
			print(div.value, "\n")

		except ElementDoesNotExist:
			print("no walmart # ", current_walmart, "\n")
			current_walmart +=1 
			continue

		current_walmart += 1

	save_file.close()
	browser.quit()

run()
