#!/usr/local/bin/python3.3

"""This module loads classifiers that run on the GPU

Created on May 14, 2015
@author: Matthew Sevrens
"""

import ctypes
import json
import logging
import scipy.stats as ss
from meerkat.various_tools import load_params

LOGSOFTMAX_THRESHOLD = -1

def get_cnn(model_name):
	"""Load a CNN model by name"""

	# Load CNN and Label map
	if model_name == "bank_merchant":
		return get_cnn_by_path(
			"meerkat/classification/models/624_class_bank_CNN.t7b",
			"meerkat/classification/label_maps/new_rlm_bank.json")
	elif model_name == "card_merchant":
		return get_cnn_by_path(
			"meerkat/classification/models/751_class_card_CNN.t7b",
			"meerkat/classification/label_maps/new_rlm_card.json")
	elif model_name == "bank_debit_subtype":
		return get_cnn_by_path(
			"meerkat/classification/models/bank_debit_subtype_CNN_01192016.t7b",
			"meerkat/classification/label_maps/bank_debit_subtype_label_map_01192016.json")
	elif model_name == "bank_credit_subtype":
		return get_cnn_by_path(
			"meerkat/classification/models/bank_credit_subtype_CNN_01192016.t7b",
			"meerkat/classification/label_maps/bank_credit_subtype_label_map_01192016.json")
	elif model_name == "card_debit_subtype":
		return get_cnn_by_path(
			"meerkat/classification/models/card_debit_subtype_CNN.t7b",
			"meerkat/classification/label_maps/card_debit_subtype_label_map.json")
	elif model_name == "card_credit_subtype":
		return get_cnn_by_path(
			"meerkat/classification/models/card_credit_subtype_CNN.t7b",
			"meerkat/classification/label_maps/card_credit_subtype_label_map.json")
	else:
		print("Requested CNN does not exist. Please reference an existing model")

#accuray, dict_path are keyword-only arguments!!!
def get_ensemble_by_path(*model_path, **kwargs):
	accuracy = kwargs.get('accuracy', [])
	dict_path = kwargs.get('dict_path','')
	model_path1, model_path2, model_path3 = model_path[:]
	ranks = ss.rankdata(accuracy)
	weight1, weight2, weight3 = [ranks[i]/(sum(ranks)+0.0) for i in range(len(ranks))]
	"""Load a function to process transactions using a CNN"""
	# pylint:disable=unused-variable
	lualib = ctypes.CDLL(
		"/home/ubuntu/torch/install/lib/libluajit.so",
		mode=ctypes.RTLD_GLOBAL)

	# Must Load Lupa After the Preceding Line
	import lupa
	# pylint:disable=no-name-in-module
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

	reverse_label_map = load_params(dict_path)
	lua_load_model = 'model1 = Model:makeCleanSequential(torch.load("' + model_path1 + '"))'
	lua.execute(lua_load_model)
	lua_load_model = 'model2 = Model:makeCleanSequential(torch.load("' + model_path2 + '"))'
	lua.execute(lua_load_model)
	lua_load_model = 'model3 = Model:makeCleanSequential(torch.load("' + model_path3 + '"))'
	lua.execute(lua_load_model)
	"""
	lua_load_model = 'model4 = Model:makeCleanSequential(torch.load("' + model_path4 + '"))'
	lua.execute(lua_load_model)
	lua_load_model = 'model5 = Model:makeCleanSequential(torch.load("' + model_path5 + '"))'
	lua.execute(lua_load_model)
	"""

	# Prepare CNN
	lua.execute('''
		model1 = model1:type("torch.CudaTensor")
		model2 = model2:type("torch.CudaTensor")
		model3 = model3:type("torch.CudaTensor")
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
			transLen = table.getn(trans)
			batch = torch.Tensor(transLen, #alphabet, 123)
			for k = 1, transLen do
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
		function(batch, weight1, weight2, weight3)
			batchLen = batch:size(1)
			batch = batch:transpose(2, 3):contiguous():type("torch.CudaTensor")
			output1 = model1:forward(batch)
			output2 = model2:forward(batch)
			output3 = model3:forward(batch)
			output = output1*weight1 + output2*weight2 + output3*weight3
			max, decision = output:double():max(2)
			labels = {}
			activations = {}
			for k = 1, batchLen do
				labels[k] = decision:select(1, k)[1]
				activations[k] = max:select(1, k)[1]
			end
			return labels, activations
		end
	''')

	# Generate Helper Function
	def apply_cnn(trans, doc_key="description", label_key="CNN"):
		"""Apply CNN to transactions"""

		trans_list = [' '.join(x[doc_key].split()) for x in trans]
		table_trans = list_to_table(trans_list)
		batch = make_batch(table_trans)
		labels, activations = process_batch(batch, weight1, weight2, weight3)
		decisions = list(labels.values())
		confidences = list(activations.values())

		for index, transaction in enumerate(trans):
			#label = reverse_label_map.get(str(decisions[index]), "") if confidences[index] > LOGSOFTMAX_THRESHOLD else ""
			label = reverse_label_map.get(str(decisions[index]), "")
			transaction[label_key] = label

		return trans

	return apply_cnn

if __name__ == "__main__":
	# pylint:disable=pointless-string-statement
	"""Print a warning to not execute this file as a module"""
	logging.warning("This module is a library that contains useful functions;"
		"it should not be run from the console.")

