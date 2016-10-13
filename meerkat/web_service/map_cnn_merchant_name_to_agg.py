import json
from meerkat.various_tools import load_params

input_json_path = "meerkat/web_service/config/CNN_Agg.txt"
output_json_path = "meerkat/web_service/config/merchant_name_map.json"

old_map = load_params(input_json_path)
new_map = {}
for item in old_map:
	cnn_name, agg_name = item["CNN_Name"], item["Agg_Name"]
	if cnn_name not in new_map:
		new_map[cnn_name] = [agg_name]
	else:
		new_map[cnn_name].append(agg_name)

with open(output_json_path, "w") as outfile:
	json.dump(new_map, outfile, sort_keys=True, indent=4, separators=(',', ': '))
print("Found {0} CNN merchants".format(len(new_map)))
