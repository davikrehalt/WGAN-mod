from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T

class Fully_Connected_Layer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,activation=T.nnet.relu):
        self.activation=activation
        self.input = input
        if W is None:
            max_value=np.sqrt(6. / (n_in + n_out))
            W_values = np.asarray(
                rng.uniform(
                    low=-max_value,
                    high=max_value,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.asarray(
                rng.uniform(
                    low=0,
                    high=1,
                    size=(n_out,)
                ),
                dtype=theano.config.floatX
            )
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
    def mse(self, y):
        # error between output and target
        return T.mean((self.output - y) ** 2)

class MLP(object):
    def __init__(self, rng, input, n_layers):
        self.input = input
        current_input=input
        self.layers=[]
        for layer_num in range(len(n_layers)-2):
            self.layers.append(Fully_Connected_Layer(
                rng=rng,
                input=current_input,
                n_in=n_layers[layer_num],
                n_out=n_layers[layer_num+1]
            ))
            current_input=self.layers[-1].output
        self.layers.append(Fully_Connected_Layer(
            rng=rng,
            input=current_input,
            n_in=n_layers[len(n_layers)-2],
            n_out=n_layers[len(n_layers)-1],
            activation=None
        ))
        self.output=self.layers[-1].output
        self.mse = self.layers[-1].mse
        self.params_W = [layer.W for layer in self.layers]
        self.params_b = [layer.b for layer in self.layers]

def generate_data_test(length):
    data_input=np.random.random((length,1))
    data_output=np.vstack([(np.sin(20*data_input[:,0])/20)]).T
    datax = theano.shared(np.asarray(data_input, dtype=theano.config.floatX), borrow=True)
    datay = theano.shared(np.asarray(data_output, dtype=theano.config.floatX), borrow=True)
    return (datax,datay)

def load_data_test():
    valid_set_x, valid_set_y = generate_data_test(1000)
    train_set_x, train_set_y = generate_data_test(10000)
    rval = [(valid_set_x, valid_set_y), (train_set_x, train_set_y)]
    return rval

def example_train(learning_rate=0.01, n_epochs=1000, batch_size=20):
    print_initial_parameters = False 
    print_end_parameters = False
    print_initial_gradient = False 
    import timeit
    datasets = load_data_test()

    train_set_x, train_set_y = datasets[1]
    valid_set_x, valid_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = np.random.RandomState(1001)
    #rng = np.random.RandomState()

    # construct the MLP class
    network = MLP(
        rng=rng,
        input=x,
        n_layers=[1,10,10,10,10,10,1]
    )
    cost = (
        network.mse(y)
    )
    if print_initial_parameters:
        print('printing initial parameters')
        for W in network.params_W:
            print(W.get_value())
        for b in network.params_b:
            print(b.get_value())
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    validate_model = theano.function(
        inputs=[index],
        outputs=network.mse(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    gparams_W = [T.grad(cost, param) for param in network.params_W]
    gparams_b = [T.grad(cost, param) for param in network.params_b]
    print_W_gradient = theano.function(
        inputs=[],
        outputs=gparams_W,
        givens={
            x: train_set_x,
            y: train_set_y
        }
    )
    print_b_gradient = theano.function(
        inputs=[],
        outputs=gparams_b,
        givens={
            x: train_set_x,
            y: train_set_y
        }
    )
    if print_initial_gradient:
        print('printing initial gradient')
        for dW in print_W_gradient():
            print(dW)
        for db in print_W_gradient():
            print(db)
    num_param_W = len(network.params_W)
    num_param_b = len(network.params_b)
    updates=[]
    for param_i in range(num_param_W):
        updates.append((network.params_W[param_i], network.params_W[param_i] - learning_rate * gparams_W[param_i]))
    for param_i in range(num_param_b):
        updates.append((network.params_b[param_i], network.params_b[param_i] - learning_rate * gparams_b[param_i]))
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('... training')

    validation_frequency = n_train_batches
    start_time = timeit.default_timer()

    epoch = 0

    while (epoch < n_epochs):
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation mean square error %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss
                    )
                )
        epoch+=1

    end_time = timeit.default_timer()
    print(('The code ran for %.2fs' % (end_time - start_time)))
    with open('test_mlp_model.pkl', 'wb') as f:
        pickle.dump(network, f)
    if print_end_parameters:
        print('printing end parameters')
        for W in network.params_W:
            print(W.get_value())
        for b in network.params_b:
            print(b.get_value())

def example_predict(length=1000):
    import matplotlib.pyplot as plt
    network = pickle.load(open('test_mlp_model.pkl','rb'))
    predict_model = theano.function(
        inputs=[network.input],
        outputs=network.output)
    test_set_x = np.reshape(np.linspace(-1,2,length),(length,1))
    predicted_values = predict_model(test_set_x)
    actual_values = np.sin(20*test_set_x)/20
    plt.plot(test_set_x,predicted_values,'ro',test_set_x,actual_values,'k')
    plt.show()

if __name__ == '__main__':
    example_train()
    example_predict()
