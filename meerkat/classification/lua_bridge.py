#!/usr/local/bin/python3.3

"""This module loads classifiers that run on the GPU

Created on May 14, 2015
@author: Matthew Sevrens
"""

import ctypes
import sys
import csv
import json

def load_label_map(filename):
	"""Load a permanent label map"""

	input_file = open(filename, encoding='utf-8')
	label_map = json.loads(input_file.read())
	input_file.close()

	return label_map

def get_CNN(model_name):
	"""Load a function to process transactions using a CNN"""

	lualib = ctypes.CDLL\
	("/home/ubuntu/torch/install/lib/libluajit.so", mode=ctypes.RTLD_GLOBAL)

	# Must Load Lupa After the Preceding Line
	import lupa
	from lupa import LuaRuntime

	# Load Runtime and Lua Modules
	lua = LuaRuntime(unpack_returned_tuples=True)
	nn = lua.require('nn')
	model = lua.require('meerkat/classification/lua/model')
	torch = lua.require('torch')
	cutorch = lua.require('cutorch')
	cunn = lua.require('cunn')
	
	# Load Config
	lua.execute('''
		dofile("meerkat/classification/lua/config.lua")
	''')

	# Load CNN and Label map
	if model_name == "bank":
		reverse_label_map = load_label_map\
		("meerkat/classification/label_maps/reverse_bank_label_map.json")
		lua.execute('''
			model = Model:makeCleanSequential(torch.load("meerkat/classification/models/612_class_bank_CNN.t7b"))
		''')
	elif model_name == "card":
		reverse_label_map = load_label_map\
		("meerkat/classification/label_maps/reverse_card_label_map.json")
		lua.execute('''
			model = Model:makeCleanSequential(torch.load("meerkat/classification/models/750_class_card_CNN.t7b"))
		''')
	else:
		print("Requested CNN does not exist. Please reference an existing model")

	# Prepare CNN
	lua.execute('''
		model = model:type("torch.CudaTensor")
		cutorch.synchronize()

		alphabet = config.alphabet
		dict = {}
		for i = 1, #alphabet do
			dict[alphabet:sub(i,i)] = i
		end
	''')

	# Load Lua Functions
	lua.execute('''
		function stringToTensor (str, l, input)
			local s = str:lower()
			local l = l or #s
			local t = input or torch.Tensor(#alphabet, l)
			t:zero()
			for i = #s, math.max(#s - l + 1, 1), -1 do
				if dict[s:sub(i,i)] then
					t[dict[s:sub(i,i)]][#s - i + 1] = 1
				end
			end
			return t
		end
	''')

	make_batch = lua.eval('''
		function(trans)
			batch = torch.Tensor(128, #alphabet, 123)
			for k = 1, 128 do
				stringToTensor(trans[k], 123, batch:select(1, k))
			end
			return batch
		end
	''')

	list_to_table = lua.eval('''
		function(trans)
			local t, i = {}, 1
			for item in python.iter(trans) do
				t[i] = item
				i = i + 1
			end
			return t
		end
	''')

	process_batch = lua.eval('''
		function(batch)
			batch = batch:transpose(2, 3):contiguous():type("torch.CudaTensor")
			output = model:forward(batch)
			max, decision = output:double():max(2)
			labels = {}
			for k = 1, 128 do
				labels[k] = decision:select(1, k)[1]
			end
			return labels
		end
	''')

	# Generate Helper Function
	def apply_CNN(trans):
		"""Apply CNN to transactions in batches of 128"""
		
		trans_list = [' '.join(x["description"].split()) for x in trans]
		table_trans = list_to_table(trans_list)
		batch = make_batch(table_trans)
		labels = process_batch(batch)
		decisions = list(labels.values())
		
		for i, t in enumerate(trans):
			t["CNN"] = reverse_label_map.get(str(decisions[i]), "")

		return trans

	return apply_CNN

if __name__ == "__main__":
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;" +\
 "it should not be run from the console.")

