#!/usr/local/bin/python3

"""Tools for Longtail

Created on Aug 9, 2016
@author: Matthew Sevrens
"""

import tensorflow as tf

from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.util import nest

state_size_with_prefix = rnn_cell._state_size_with_prefix

def infer_state_dtype(explicit_dtype, state):

	if explicit_dtype is not None:
		return explicit_dtype
	elif nest.is_sequence(state):

		inferred_dtypes = [element.dtype for element in nest.flatten(state)]

	    if not inferred_dtypes:
			raise ValueError("Unable to infer dtype from empty state.")

		all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])

		if not all_same:
			raise ValueError(
				"State has tensors of different inferred_dtypes. Unable to infer a "
				"single representative dtype.")

		return inferred_dtypes[0]
	else:
		return state.dtype

def rnn_step(time, sequence_length, min_sequence_length, max_sequence_length,
			zero_output, state, call_cell, state_size, skip_conditionals=False):

def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
				dtype=None, parallel_iterations=None, swap_memory=False,
				time_major=False, scope=None):

	if not isinstance(cell, rnn_cell.RNNCell):
		raise TypeError("cell must be an instance of RNNCell")

	parallel_iterations = parallel_iterations or 32
	flat_input = nest.flatten(inputs)

	if not time_major:
		flat_input = tuple(tf.transpose(fi, perm=[1,0,2]) for fi in flat_input)

	if sequence_length is not None:
		sequence_length = tf.to_int32(sequence_length)
		sequence_length = tf.identity(sequence_length, name="sequence_length")

	# Create a New Scope
	with vs.variable_scope(scope or "RNN") as varscope:

		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		input_shape = tuple(tf.shape(fi) for fi in flat_input)
		batch_size = input_shape[0][1]

		# Validate Batch Size
		for input_ in input_shape:
			if input_[1].get_shape() != batch_size.get_shape():
				raise ValueError("All inputs should have the same batch size")

		# Set Initial State
		if initial_state is not None:
			state = initial_state
		else:
			if not dtype:
				raise ValueError("If no initial_state is provided, dtype must be.")
			state = cell.zero_state(batch_size, dtype)

		def assert_has_shape(x, shape):
			x_shape = tf.shape(x)
			packed_shape = tf.pack(shape)
			return logging_ops.Assert(
				math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
				["Expected shape for Tensor %s is " % x.name,
				packed_shape, " but saw shape: ", x_shape])

		# Perform Sequence Length Shape Validation
		if sequence_length is not None:
			with ops.control_dependencies([assert_has_shape(sequence_length, [batch_size])]):
				sequence_length = tf.identity(sequence_length, name="CheckSeqLen")

		inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

		# Run Through RNN
		(outputs, final_state) = dynamic_rnn_loop(
			cell,
			inputs,
			state,
			parallel_iterations=parallel_iterations,
			swap_memory=swap_memory,
			sequence_length=sequence_length,
			dtype=dtype)

		# Transpose Output Back to Shape [Batch, Time, Depth]
		if not time_major:
			flat_output = nest.flatten(outputs)
			flat_output = [tf.transpose(output, perm=[1,0,2]) for output in flat_output]
			outputs = nest.pack_sequence_as(structure=outputs, flat_sequence=flat_output)

		return (outputs, final_state)

def dynamic_rnn_loop(cell, inputs, initial_state, parallel_iterations,
					swap_memory, sequence_length=None, dtype=None):

	state = initial_state
	assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

	state_size = cell.state_size

	flat_input = nest.flatten(inputs)
	flat_output_size = nest.flatten(cell.output_size)

	# Construct an Initial Output
	input_shape = array_ops.shape(flat_input[0])
	time_steps = input_shape[0]
	batch_size = input_shape[1]

	inputs_got_shape = tuple(fi.get_shape().with_rank_at_least(3) for fi in flat_input)

	const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

	# Validate Input Shapes
	for shape in inputs_got_shape:

		if not shape[2:].is_fully_defined():
			raise ValueError(
				"Input size (depth of inputs) must be accessible via shape inference,"
				" but saw value None.")

		got_time_steps = shape[0]
		got_batch_size = shape[1]

		if const_time_steps != got_time_steps:
			raise ValueError(
				"Time steps is not the same for all the elements in the input in a "
				"batch.")

		if const_batch_size != got_batch_size:
			raise ValueError(
				"Batch_size is not the same for all the elements in the input.")

	# Prepare Dynamic Conditional Copying of State & Output
	def create_zero_arrays(size):
		size = state_size_with_prefix(size, prefix=[batch_size])
		return tf.zeros(tf.pack(size), infer_state_dtype(dtype, state))

	flat_zero_output = tuple(create_zero_arrays(output) for output in flat_output_size)
	zero_output = nest.pack_sequence_as(structure=cell.output_size, flat_sequence=flat_zero_output)

	if sequence_length is not None:
		min_sequence_length = math_ops.reduce_min(sequence_length)
		max_sequence_length = math_ops.reduce_max(sequence_length)

	time = tf.constant(0, dtype=dtypes.int32, name="time")

	with ops.op_scope([], "dynamic_rnn") as scope:
		base_name = scope

	# Helper function to Create Tensor Arrays
	def create_ta(name, dtype):
		return tensor_array_ops.TensorArray(dtype=dtype, size=time_steps, tensor_array_name=base_name + name)

	output_ta = tuple(create_ta("output_%d" % i, infer_state_dtype(dtype, state)) for i in range(len(flat_output_size)))
	input_ta = tuple(create_ta("input_%d" % i, flat_input[0].dtype) for i in range(len(flat_input)))
	input_ta = tuple(ta.unpack(input_) for ta, input_ in zip(input_ta, flat_input))

	# Function to Perform at Each Time Step
	def time_step(time, output_ta, state):

		input_t = tuple(ta.read(time) for ta in input_ta)

		# Restore Shape Information
		for input_, shape in zip(input_t, inputs_got_shape):
			input_.set_shape(shape[1:])

		input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
		call_cell = lambda: cell(input_t, state)

		if sequence_length is not None:
			(output, new_state) = rnn_step(
				time=time,
				sequence_length=sequence_length,
				min_sequence_length=min_sequence_length,
				max_sequence_length=max_sequence_length,
				zero_output=zero_output,
				state=state,
				call_cell=call_cell,
				state_size=state_size,
				skip_conditionals=False)
		else:
			(output, new_state) = call_cell()

		# Pack State if Using State Tuples
		output = nest.flatten(output)

		output_ta = tuple(ta.write(time, out) for ta, out in zip(output_ta, output))

		return (time + 1, output_ta, new_state)

	# Process Over Each Time Step
	_, output_final_ta, final_state = tf.while_loop(
		cond=lambda time, *_: time < time_steps,
		body=_time_step,
		loop_vars=(time, output_ta, state),
		parallel_iterations=parallel_iterations,
		swap_memory=swap_memory)

	# Unpack Final Output if Not Using Output Tuples
	final_outputs = tuple(ta.pack() for ta in output_final_ta)

	# Restore Shape Information
	for output, output_size in zip(final_outputs, flat_output_size):
		shape = state_size_with_prefix(output_size, prefix=[const_time_steps, const_batch_size])
    	output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(structure=cell.output_size, flat_sequence=final_outputs)

	return (final_outputs, final_state)

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
							initial_state_fw=None, initial_state_bw=None,
							dtype=None, parallel_iterations=None,
							swap_memory=False, time_major=False, scope=None):

	if not isinstance(cell_fw, rnn_cell.RNNCell):
		raise TypeError("cell_fw must be an instance of RNNCell")
	if not isinstance(cell_bw, rnn_cell.RNNCell):
		raise TypeError("cell_bw must be an instance of RNNCell")

	if scope is None:
		name = "BiRNN"
	elif isinstance(scope, six.string_types):
		name = scope
	elif isinstance(scope, vs.VariableScope):
		name = scope.name
	else:
		raise TypeError("scope must be a string or an instance of VariableScope")

	# Forward Direction
	with vs.variable_scope(name + "_FW") as fw_scope:

    	output_fw, output_state_fw = dynamic_rnn(
		cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
		initial_state=initial_state_fw, dtype=dtype,
		parallel_iterations=parallel_iterations, swap_memory=swap_memory,
		time_major=time_major, scope=fw_scope)

	# Backward direction
	if not time_major:
		time_dim = 1
		batch_dim = 0
	else:
		time_dim = 0
		batch_dim = 1

	with vs.variable_scope(name + "_BW") as bw_scope:

		inputs_reverse = array_ops.reverse_sequence(
			input=inputs, seq_lengths=sequence_length,
			seq_dim=time_dim, batch_dim=batch_dim)

		tmp, output_state_bw = dynamic_rnn(
			cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
			initial_state=initial_state_bw, dtype=dtype,
			parallel_iterations=parallel_iterations, swap_memory=swap_memory,
			time_major=time_major, scope=bw_scope)

		output_bw = array_ops.reverse_sequence(
			input=tmp, seq_lengths=sequence_length,
			seq_dim=time_dim, batch_dim=batch_dim)

	outputs = (output_fw, output_bw)
	output_states = (output_state_fw, output_state_bw)

	return (outputs, output_states)