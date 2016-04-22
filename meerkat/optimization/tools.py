def add_local_params(params):
	"""Adds additional local params"""
	params["mode"] = "train"
	params["optimization"] = {}
	params["optimization"]["scores"] = []

	params["optimization"]["settings"] = {
		"folds": 1,
		"initial_search_space": 25,
		"initial_learning_rate": 0.25,
		"iteration_search_space": 15,
		"iteration_learning_rate": 0.1,
		"gradient_descent_iterations": 10,
		"max_predictive_accuracy": 97.5,
		"min_recall": 31,
		"min_percent_labeled": 31
	}
	return params

if __name__ == "__main__":
	msg = "This is a library of useful functions, do not run it from the command line."
	logging.critical(msg)




