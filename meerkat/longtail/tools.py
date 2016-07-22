"""The place where you put frequently used longtail handler dedicated functions"""

def get_tensor(graph, name):
	"""Get tensor by name"""
	return graph.get_tensor_by_name(name)

def get_op(graph, name):
	"""Get operation by name"""
	return graph.get_operation_by_name(name)

def get_variable(graph, name):
	"""Get variable by name"""
	with graph.as_default():
		variable = [v for v in tf.all_variables() if v.name == name][0]
		return variable