from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from optimize import rmsprop
from ops import tmax,tmin
from theano.tensor.shared_randomstreams import RandomStreams
from lmlp import LMLP

class Simple_Generator(object):
    def __init__(self,rng,input_rand,g_shape,r_shape):
        self.input=input_rand
        self.g_shape=g_shape
        self.r_shape=r_shape
        self.generator = LMLP(
            rng=rng,
            input=input_rand,
            info_layers=g_shape,
            init=2
        )
        self.output=self.generator.output
        self.reversor = LMLP(
            rng=rng,
            input=self.output,
            info_layers=r_shape
        ) 
        self.g_params=self.generator.params
        self.r_params=self.reversor.params
        self.mse = self.reversor.mse
        self.gradient_cost=self.reversor.gradient_cost
        self.max_gradient = self.reversor.max_gradient

def example_train(n_epochs=1000,batch_size=200):
    import timeit
    g_shape=[(5,1,20),(5,20,20),(5,20,2)]
    r_shape=[(5,2,20),(5,20,20),(5,20,1)]
    g_per_epoch=20
    r_per_epoch=10
    plot_time=10
    print_validation_g_parameters = False
    print_validation_r_parameters = False
    initial_reversor_train=1000
    print('... building the model')
    rng = np.random.RandomState(1001)
    trng = RandomStreams(seed=234)
    x_rand = T.matrix('x_rand')

    generator = Simple_Generator(
        rng=rng,
        input_rand=x_rand,
        g_shape=g_shape,
        r_shape=r_shape
    )
        
    r_cost = generator.mse(x_rand)+1.0/(1.0-generator.gradient_cost)
    r_updates = rmsprop(r_cost,generator.r_params)

    f = lambda x:((1.0-T.dot(x**2,np.array([[1.0],[1.0]])))**2).mean()
    cost = f(generator.output)

    g_cost = generator.mse(x_rand)+cost
    g_updates = rmsprop(g_cost,generator.g_params)

    train_reversor = theano.function(
        inputs=[],
        outputs=r_cost,
        updates=r_updates,
        givens={
            x_rand : trng.uniform(
                size=(batch_size, g_shape[0][1]), 
                low=-1.0,
                high=1.0
            )
        }
    )

    train_generator = theano.function(
        inputs=[],
        outputs=g_cost,
        updates=g_updates,
        givens={
            x_rand : trng.uniform(
                size=(batch_size, g_shape[0][1]), 
                low=-1.0,
                high=1.0
            )
        }
    )
    test_generator = theano.function(
        inputs=[],
        outputs=cost,
        givens={
            x_rand : trng.uniform(
                size=(batch_size, g_shape[0][1]), 
                low=-1.0,
                high=1.0
            )
        }
    )
    test_reversor = theano.function(
        inputs=[],
        outputs=generator.mse(x_rand),
        givens={
            x_rand : trng.uniform(
                size=(batch_size, g_shape[0][1]), 
                low=-1.0,
                high=1.0
            )
        }
    )
    get_max_gradient = theano.function(
        inputs=[],
        outputs=generator.max_gradient
    )

    print('... training')
    start_time = timeit.default_timer()
    for _ in range(initial_reversor_train):
        _ = train_reversor()
    for epoch in range(n_epochs):
        if epoch % plot_time == 0:
            if print_validation_g_parameters:
                print('printing gradient parameters')
                for param in generator.g_params:
                    print(param.get_value())
            if print_validation_r_parameters:
                print('printing reversor parameters')
                for param in reversor.r_params:
                    print(param.get_value())
            with open('test_generator_model.pkl', 'wb') as f:
                pickle.dump(generator.generator, f)
            with open('test_reversor_model.pkl', 'wb') as f:
                pickle.dump(generator.reversor, f)
            example_graph()

        for _ in range(g_per_epoch):
            _ = train_generator()
                
        for _ in range(r_per_epoch):
            _ = train_reversor()

        gen_cost=test_generator()
        rev_cost=test_reversor()
        max_grad=get_max_gradient()
        print('epoch %i, generator cost %f, reversor cost %f, max gradient %f' % (
            epoch,gen_cost,rev_cost,max_grad))
        
    end_time = timeit.default_timer()
    print(('The code ran for %.2fs' % (end_time - start_time)))

def example_graph(length=1000):
    import matplotlib.pyplot as plt
    generator = pickle.load(open('test_generator_model.pkl','rb'))
    reversor = pickle.load(open('test_reversor_model.pkl','rb'))
    gen_model = theano.function(
        inputs=[generator.input],
        outputs=generator.output
    )
    rev_model = theano.function(
        inputs=[reversor.input],
        outputs=reversor.output
    )
    data_input_g = np.reshape(np.linspace(-1.0,1.0,length),(length,1))
    graph_g = gen_model(data_input_g)
    z=np.linspace(-1.0,1.0,length)
    plt.subplot(3, 1, 1)
    plt.plot(z,graph_g[:,0])
    plt.ylabel('x comp')
    plt.subplot(3, 1, 2)
    plt.plot(z,graph_g[:,1])
    plt.ylabel('y comp')
    plt.subplot(3, 1, 3)
    rev_graph = rev_model(graph_g)
    plt.plot(z,rev_graph)
    plt.ylabel('reversor')
    plt.show()


if __name__ == "__main__":
    example_train()
