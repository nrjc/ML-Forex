from fann2 import libfann

learning_rate = 0.3
num_layers = 3
num_input = 4
num_hidden = 4
num_output = 1

desired_error = 0.0001
max_iterations = 7
iterations_between_reports = 1

ann = libfann.neural_net()
ann.create_standard_array((4,4,1))
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file("EURUSDTESTTEST", max_iterations, iterations_between_reports, desired_error)

ann.save("new.net")
