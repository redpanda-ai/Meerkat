from splinter.browser import Browser
from splinter.exceptions import ElementDoesNotExist
from pprint import pprint
import os.path

browser = Browser()
base_url = "http://www.mccalifornia.com/"
current_mcdonalds = 3
found = []

while current_mcdonalds < 129830:

	browser.visit(base_url + str(current_mcdonalds))
	new = {
		"internal_store_number" : ("F" + str(current_mcdonalds)),
		"address" : ""
	}

	try:
		elements = browser.find_by_css("li.address_1 h3, li.address_2 h3, li.address_3 h3")
		div = elements[0]
		new["address"] = div.value
		found.append(new)

		# Save Result
		save_file = open("mcdonalds_store_numbers.txt", "a")
		pprint(new, save_file)
		save_file.close()


		print(str(current_mcdonalds), ":")
		print(div.value, "\n")
	except ElementDoesNotExist:
		print("no mcdonalds # ", current_mcdonalds)
		current_mcdonalds +=1 
		continue

	current_mcdonalds += 1

browser.quit()