from splinter.browser import Browser
from splinter.exceptions import ElementDoesNotExist
from pprint import pprint
import os.path

def run():
	browser = Browser()
	base_url = "http://www.mccalifornia.com/"
	current_mcdonalds = 30000
	found = []
	save_file = open("../../data/misc/Store\ Numbers/Clean/mcdonalds_store_numbers.txt", "a")

	# Clear Contents of Old Run
	#open('mcdonalds_store_numbers.txt', 'w').close()

	while current_mcdonalds < 40000:

		browser.visit(base_url + str(current_mcdonalds))

		try:

			# Find Adress on Page
			elements = browser.find_by_css("li.address_1 h3, li.address_2 h3, li.address_3 h3, li.address_4 h3, li.address_5 h3")
			div = elements[0]

			# Parse
			lines = div.value.split('\n')
			
			if len(lines) == 2:
				address, extra = lines
			elif len(lines) == 3:
				address, trash, extra = lines


			city, extra = extra.split(',')
			state, zip_code = extra.lstrip(' ').split(' ')

			# Format
			new = {
				"internal_store_number" : ("F" + str(current_mcdonalds)),
				"address" : address,
				"city" : city,
				"state" : state,
				"zip_code" : zip_code
			}

			found.append(new)

			# Save Result
			pprint(new, save_file)

			# Print
			print(str(current_mcdonalds), ":")
			print(div.value, "\n")

		except ElementDoesNotExist:
			print("no mcdonalds # ", current_mcdonalds, "\n")
			current_mcdonalds +=1 
			continue

		current_mcdonalds += 1

	save_file.close()
	browser.quit()

run()