import numpy
import scipy

class neural_network:
    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes, learning_rate):
        self.weight_input_hidden = numpy.random.normal(0.0, 1, (num_hidden_nodes, num_input_nodes))
        self.weight_hidden_output = numpy.random.normal(0.0, 1, (num_output_nodes, num_input_nodes))

        self.learning_rate = learning_rate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def fit(self, train_images, train_targets, epoches:int, print_plt:bool=False):
        # required .T because numpy.dot with weight
        images = numpy.array(train_images, ndmin=2).T
        targets = numpy.array(train_targets, ndmin=2).T

        for i in range(epoches):
            # dot product of two arrays
            net_hidden = numpy.dot(self.weight_input_hidden, images)
            # using sigmoid
            activation_neurons_hidden = self.activation_function(net_hidden)

            net_output = numpy.dot(self.weight_hidden_output, activation_neurons_hidden)
            activation_neurons_output = self.activation_function(net_output)

            output_errors = targets - activation_neurons_output
            # hidden layer's error is redistributed by the output errors of the weight ratio
            hidden_errors = numpy.dot(self.weight_hidden_output.T, output_errors)

            # error back propagation using chain rule
            self.weight_hidden_output \
                += self.learning_rate * numpy.dot((output_errors * activation_neurons_output
                                                   * (1.0 - activation_neurons_output)), activation_neurons_hidden.T)
            self.weight_input_hidden \
                += self.learning_rate * numpy.dot((hidden_errors * activation_neurons_hidden
                                                   * (1.0 - activation_neurons_hidden)), images.T)

        pass
