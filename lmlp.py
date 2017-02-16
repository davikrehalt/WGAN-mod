from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from optimize import rmsprop
from ops import tmax,tmin

class Lipshitz_Layer(object):
    def __init__(self, rng, input, n_max, n_in,n_out, W=None, b=None,init=0):
        self.input = input
        if W is None:
            if init == 0:
                min_value = -1.0 / n_in
                max_value = 1.0 / n_in
            elif init == 1:
                min_value = -np.sqrt(3.0/n_in)
                max_value = np.sqrt(3.0/n_in)
            elif init == 2:
                min_value = 0.5 / n_in
                max_value = 1.5 / n_in
            W_values = np.asarray(
                rng.uniform(
                    low=min_value,
                    high=max_value,
                    size=(n_max, n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.asarray(
                rng.uniform(
                    low=-0.5,
                    high=0.5,
                    size=(n_max,n_out)
                ),
                dtype=theano.config.floatX
            )
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params=[self.W,self.b]
        self.output = (T.dot(self.input, self.W) + self.b).max(axis=1)
        self.gradient_norms=abs(self.W).sum(axis=1)
        self.gradient_cost=T.sum(tmax(self.gradient_norms-1.0,0.0))
        self.max_gradient=T.max(self.gradient_norms)

class LMLP(object):
    def __init__(self, rng, input, info_layers,params=None,init=0):
        #info_layer has one entry per layer
        #n_max,n_in,n_out
        self.input = input
        current_input=self.input
        self.layers=[]
        self.gradient_cost=0.0
        self.max_gradient=1.0
        if params is None:
            for info in info_layers:
                self.layers.append(Lipshitz_Layer(
                    rng=rng,
                    input=current_input,
                    n_max=info[0],
                    n_in=info[1],
                    n_out=info[2],
                    init=init
                ))
                current_input=self.layers[-1].output
                self.max_gradient*=self.layers[-1].max_gradient
                self.gradient_cost+=self.layers[-1].gradient_cost
        else:
            index = 0
            for info in info_layers:
                self.layers.append(Lipshitz_Layer(
                    rng=rng,
                    input=current_input,
                    n_max=info[0],
                    n_in=info[1],
                    n_out=info[2],
                    W=params[index],
                    b=params[index+1]
                ))
                index+=2
                current_input=self.layers[-1].output
                self.max_gradient*=self.layers[-1].max_gradient
                self.gradient_cost+=self.layers[-1].gradient_cost
                
        self.output=self.layers[-1].output
        self.params = [param for layer in self.layers for param in layer.params]
    def mse(self,y):
        return T.mean((self.output - y) ** 2)

def generate_data_test(length,data_num):
    data_input=2*np.random.random((length,1))-1
    if data_num == 0:
        data_output=np.vstack([(5*np.sin(5*data_input[:,0]))]).T
    elif data_num == 1:
        data_output=np.vstack([(np.sin(20*data_input[:,0])/20)]).T
    elif data_num == 2:
        data_output=np.vstack([(np.sin(5*data_input[:,0])/5)]).T
    elif data_num == 3:
        data_output=np.vstack([np.exp(data_input[:,0]-1.0)]).T
    datax = theano.shared(np.asarray(data_input, dtype=theano.config.floatX), borrow=True)
    datay = theano.shared(np.asarray(data_output, dtype=theano.config.floatX), borrow=True)
    return (datax,datay)

def load_data_test(data_num):
    valid_set_x, valid_set_y = generate_data_test(1000,data_num)
    train_set_x, train_set_y = generate_data_test(10000,data_num)
    rval = [(valid_set_x, valid_set_y), (train_set_x, train_set_y)]
    return rval

def example_train(n_epochs=1000, batch_size=20,gradient_reg=1.0,data_num=2):
    print_initial_parameters = False
    print_initial_gradient = False 
    print_initial_gradient_norms = False
    print_validation_parameters = False
    print_end_parameters = False
    plot_time=100

    import timeit
    datasets = load_data_test(data_num)

    train_set_x, train_set_y = datasets[1]
    valid_set_x, valid_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar() 
    x = T.matrix('x') 
    y = T.matrix('y')  

    rng = np.random.RandomState(1001)

    # construct the MLP class
    network = LMLP(
        rng=rng,
        input=x,
        info_layers=[(5,1,20),(5,20,20),(5,20,20),(5,20,1)]
    )
    cost = (network.mse(y)+gradient_reg/(1.0-network.gradient_cost))
    if print_initial_parameters:
        print('printing initial parameters')
        for param in network.params:
            print(param.get_value())
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
    gparams = [T.grad(cost, param) for param in network.params]
    get_gradient = theano.function(
        inputs=[],
        outputs=gparams,
        givens={
            x: train_set_x,
            y: train_set_y
        }
    )
    get_gradient_cost = theano.function(
        inputs=[],
        outputs=network.gradient_cost,
        givens={
        }
    )
    get_gradient_norms = theano.function(
        inputs=[],
        outputs=[layer.gradient_norms for layer in network.layers],
        givens={
        }
    )
    get_gradient_max = theano.function(
        inputs=[],
        outputs=network.max_gradient,
        givens={
        }
    )
    if print_initial_gradient_norms:
        print('printing gradient norms')
        for matrix in get_gradient_norms():
            print(matrix)

    if print_initial_gradient:
        print('printing initial gradient')
        for item in get_gradient():
            print(item)
    num_params = len(network.params)
    updates=rmsprop(cost,network.params)
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
    plot_frequency = n_train_batches*plot_time
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
                this_gradient_max = get_gradient_max()

                print(
                    'epoch %i, minibatch %i/%i, validation mean square error %f, gradient max %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss,
                        this_gradient_max
                    )
                )
            if (iter + 1) % plot_frequency == 0:
                if print_validation_parameters:
                    print('printing parameters')
                    for param in network.params:
                        print(param.get_value())
                with open('test_mlp_model.pkl', 'wb') as f:
                    pickle.dump(network, f)
                example_predict(1000,data_num)
        epoch+=1

    end_time = timeit.default_timer()
    print(('The code ran for %.2fs' % (end_time - start_time)))
    if print_end_parameters:
        print('printing end parameters')
        for param in network.params:
            print(param.get_value())

def example_predict(length,data_num):
    import matplotlib.pyplot as plt
    network = pickle.load(open('test_mlp_model.pkl','rb'))
    predict_model = theano.function(
        inputs=[network.input],
        outputs=network.output)
    data_input = np.reshape(np.linspace(-2,2,length),(length,1))
    predicted_values = predict_model(data_input)
    if data_num == 0:
        data_output=np.vstack([(5*np.sin(5*data_input[:,0]))]).T
    elif data_num == 1:
        data_output=np.vstack([(np.sin(20*data_input[:,0])/20)]).T
    elif data_num == 2:
        data_output=np.vstack([(np.sin(5*data_input[:,0])/5)]).T
    elif data_num == 3:
        data_output=np.vstack([np.exp(data_input[:,0]-1.0)]).T
    plt.plot(data_input,predicted_values,'ro',data_input,data_output,'k')
    plt.show()

if __name__ == '__main__':
    example_train(data_num=1)
    example_predict(1000,0)
