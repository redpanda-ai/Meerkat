"""Generate a dictionary for mapping CNN merchant name to a list of Agg merchant names"""
import json
from meerkat.various_tools import load_params

def main_program():
	""""Main program"""
	input_json_path = "meerkat/web_service/config/merchantlookup.json"
	output_json_path = "meerkat/web_service/config/cnn_to_substr.json"

	old_map = load_params(input_json_path)
	new_map = {}
	for item in old_map:
		cnn_name, keyword = item["CNN Name"], item["Keyword"]
		if cnn_name not in new_map:
			new_map[cnn_name] = [keyword]
		else:
			new_map[cnn_name].append(keyword)

	with open(output_json_path, "w") as outfile:
		json.dump(new_map, outfile, sort_keys=True, indent=4, separators=(',', ': '))
	print("Found {0} CNN merchants".format(len(new_map)))

if __name__ == '__main__':
	main_program()
