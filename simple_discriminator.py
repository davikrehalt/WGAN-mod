from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from optimize import rmsprop
from lmlp import LMLP

class Simple_Discriminator(object):
    def __init__(self,rng,input_fake,input_real,info_layers):
        self.input=input_fake
        self.input_r=input_real
        self.network1=LMLP(rng,self.input,info_layers)
        self.layers=self.network1.layers
        self.params=self.network1.params
        self.output=self.network1.output
        self.gradient_cost=self.network1.gradient_cost
        self.max_gradient=self.network1.max_gradient
        self.network2=LMLP(rng,self.input_r,
                           info_layers,params=self.params)
        self.output_r=self.network2.output
        self.mean_difference=(self.output-self.output_r).mean()

def generate_data(length,data_num):
    if data_num == 0:
        prearray=np.asarray(2*np.random.random(length)-1.0,
                       dtype=theano.config.floatX)
        return theano.shared(np.reshape(prearray,(length,1)),borrow=True)

    elif data_num == 1:
        prearray=np.asarray(np.random.normal(0,1,length),
                       dtype=theano.config.floatX)
        return theano.shared(np.reshape(prearray,(length,1)),borrow=True)
    elif data_num == 2:
        prearray=np.asarray(np.random.normal(-1,1,length),
                       dtype=theano.config.floatX)
        return theano.shared(np.reshape(prearray,(length,1)),borrow=True)
    elif data_num == 3:
        prearray=np.asarray(np.random.normal(0,0.1,length),
                       dtype=theano.config.floatX)
        return theano.shared(np.reshape(prearray,(length,1)),borrow=True)
def example_train(n_epochs=100, batch_size=20,gradient_reg=1.0):
    import timeit
    print_initial_parameters    = False
    print_initial_gradient_cost = False
    print_initial_gradient_norms = False
    plot_time=10
    
    fake_x_data = generate_data(10000,0)
    real_x_data = generate_data(10000,3)
    fake_x_valid = generate_data(1000,0)
    real_x_valid = generate_data(1000,3)
    index = T.lscalar() 
    x_fake = T.matrix('x_f') 
    x_real = T.matrix('x_r') 

    n_train_batches = fake_x_data.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = fake_x_valid.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')
    rng = np.random.RandomState(1000)
    network = Simple_Discriminator(
        rng=rng,
        input_fake=x_fake,
        input_real=x_real,
        #info_layers=[(5,1,20),(5,20,20),(5,20,20),(5,20,1)]
        info_layers=[(5,1,20),(1,20,1)]
    )
    cost = -network.mean_difference+gradient_reg/(1.0-network.gradient_cost)

    if print_initial_parameters:
        print('printing initial parameters')
        for param in network.params:
            print(param.get_value())

    get_max_gradient = theano.function(
        inputs=[],
        outputs=network.max_gradient,
        givens={
        }
    )

    get_gradient_norms = theano.function(
        inputs=[],
        outputs=[layer.gradient_norms for layer in network.layers],
        givens={
        }
    )
    get_gradient_cost = theano.function(
        inputs=[],
        outputs=network.gradient_cost,
        givens={
        }
    )

    if print_initial_gradient_cost:
        print('initial gradient cost: %f '% get_gradient_cost())
    if print_initial_gradient_norms:
        print('printing gradient norms')
        for matrix in get_gradient_norms():
            print(matrix)
    validate_model = theano.function(
        inputs=[index],
        outputs=network.mean_difference,
        givens={
            x_fake: fake_x_valid[index * batch_size:(index + 1) * batch_size],
            x_real: real_x_valid[index * batch_size:(index + 1) * batch_size]
        }
    )
    updates=rmsprop(cost,network.params)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x_fake: fake_x_data[index*batch_size:(index+1)*batch_size],
            x_real: real_x_data[index*batch_size:(index+1)*batch_size]
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
                this_gradient_max = get_max_gradient()

                print(
                    'epoch %i, minibatch %i/%i, validation mean square error %f, max_gradient %f' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss,
                        this_gradient_max
                    )
                )
            if (iter + 1) % plot_frequency == 0:
                with open('test_discriminator_model.pkl', 'wb') as f:
                    pickle.dump(network, f)
                example_graph()
        epoch+=1

    end_time = timeit.default_timer()
    print(('The code ran for %.2fs' % (end_time - start_time)))
    if print_end_parameters:
        print('printing end parameters')
        for param in network.params:
            print(param.get_value())

def example_graph(length=1000):
    import matplotlib.pyplot as plt
    network = pickle.load(open('test_discriminator_model.pkl','rb'))
    predict_model = theano.function(
        inputs=[network.input],
        outputs=network.output)
    data_input = np.reshape(np.linspace(-2,2,length),(length,1))
    predicted_values = predict_model(data_input)
    uniform_graph = data_input*0.+0.5
    normal_graph_1 = np.exp(-data_input**2/2.0)/np.sqrt(2*np.pi)
    normal_graph_2 = np.exp(-(1+data_input)**2/2.0)/np.sqrt(2*np.pi)
    normal_graph_3 = np.exp(-(10*data_input)**2/2.0)/(0.1*np.sqrt(2*np.pi))
    plt.plot(data_input,predicted_values)
    plt.plot(data_input,uniform_graph)
    plt.plot(data_input,normal_graph_3)
    plt.show()

if __name__ == "__main__":
    example_train()
