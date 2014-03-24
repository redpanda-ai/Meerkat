from splinter.browser import Browser
from splinter.exceptions import ElementDoesNotExist
from pprint import pprint
import os.path

browser = Browser()
base_url = "http://www.walmart.com/storeLocator/ca_storefinder_details_short.do?edit_object_id="
current_walmart = 1
found = []

while current_walmart < 5000:

	text_number = str(current_walmart)
	while len(text_number) < 4:
		text_number = "0" + text_number

	browser.visit(base_url + str(current_walmart))
	new = {
		"internal_store_number" : ("#" + text_number),
		"address" : ""
	}

	try:
		elements = browser.find_by_css(".StoreAddress")
		div = elements[0]
		new["address"] = div.value
		found.append(new)

		# Save Result
		save_file = open("walmart_store_numbers.txt", "a")
		pprint(new, save_file)
		save_file.close()


		print(str(current_walmart), ":")
		print(div.value, "\n")
	except ElementDoesNotExist:
		print("no walmart # ", current_walmart)
		current_walmart +=1 
		continue

	current_walmart += 1

browser.quit()