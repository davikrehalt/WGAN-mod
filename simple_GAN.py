from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from optimize import rmsprop
from lmlp import LMLP
from simple_discriminator import Simple_Discriminator
from simple_generator import Simple_Generator
from ops import tmax,tmin
from numpy import sqrt
from theano.tensor.shared_randomstreams import RandomStreams

def sample_data(num):
    return_list=[]
    for _ in range(num):
        s=np.random.random()
        if s<1/3:
            mat=(np.array([[1.0,0.0],[0.0,1.0]]))
        elif s<2/3:
            mat=(np.array([[-0.5,sqrt(3)/2],[-sqrt(3)/2,-0.5]]))
        else:
            mat=(np.array([[-0.5,-sqrt(3)/2],[sqrt(3)/2,-0.5]]))
        pos=np.array([1.0,0.0])+np.random.normal(0.0,0.1,size=2)
        return_list.append(np.dot(pos,mat))
    
    return theano.shared(np.asarray(return_list,dtype=theano.config.floatX), borrow=True)
    
def example_train(data,n_epochs=100,batch_size=20):
    import timeit

    d_shape=[(5,2,20),(5,20,20),(5,20,1)]
    g_shape=[(5,1,20),(5,20,20),(5,20,2)]
    r_shape=[(5,2,20),(5,20,20),(5,20,1)]

    validation_frequency = 1
    plot_time=10
    initial_r_train=20

    g_per_epoch=5
    r_per_epoch=9

    data_train=data[0]
    data_valid=data[1]

    n_train_batches = data_train.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = data_train.get_value(borrow=True).shape[0] // batch_size

    rng = np.random.RandomState(1001)
    trng = RandomStreams(seed=234)
    print('... building the model')

    index = T.lscalar() 
    x_rand = T.matrix('x_rand')
    x_real = T.matrix('x_real') 
    x_fake = T.matrix('x_rake') 

    generator = Simple_Generator(
        rng=rng,
        input_rand=x_rand,
        g_shape=g_shape,
        r_shape=r_shape
    )
    discriminator = Simple_Discriminator(
        rng=rng,
        input_fake=generator.output,
        input_real=x_real,
        info_layers=d_shape
    )

    #the 1.0 befor the / is a hyperparameter
    g_cost = generator.mse(x_rand)+(discriminator.output).mean()
    g_updates = rmsprop(g_cost,generator.g_params)

    r_cost = generator.mse(x_rand)+1.0/(1.0-generator.gradient_cost)
    r_updates=rmsprop(r_cost,generator.r_params)
    
    d_cost = -discriminator.mean_difference+1.0/(1.0-discriminator.gradient_cost)
    d_updates = rmsprop(d_cost,discriminator.params)

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
    train_discriminator = theano.function(
        inputs=[index],
        outputs=d_cost,
        updates=d_updates,
        givens={
            x_real: data_train[index*batch_size:(index+1)*batch_size],
            x_rand : trng.uniform(
                size=(batch_size, g_shape[0][1]), 
                low=-1.0,
                high=1.0
            )
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=discriminator.mean_difference,
        givens={
            x_real: data_valid[index * batch_size:(index + 1) * batch_size],
            x_rand : trng.uniform(
                size=(batch_size, g_shape[0][1]), 
                low=-1.0,
                high=1.0
            )
        }
    )

    get_max_gradient = theano.function(
        inputs=[],
        outputs=(generator.max_gradient,discriminator.max_gradient),
        givens={
        }
    )
    get_reversor_error = theano.function(
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
    print('... training')

    start_time = timeit.default_timer()

    for _ in range(initial_r_train):
        _ = train_reversor()
    for epoch in range(n_epochs):
        for minibatch_index in range(n_train_batches):
            _ = train_discriminator(minibatch_index)
        for _ in range(g_per_epoch):
            _ = train_generator()
        for _ in range(r_per_epoch):
            _ = train_reversor()
                
        if epoch % validation_frequency == 0:
            validation_losses = [validate_model(i) for i
                                    in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            max_grad = get_max_gradient()

            print('epoch %i, mean difference %f, r_error %f,r_grad %f, d_grad %f' %
                (
                    epoch,
                    this_validation_loss,
                    get_reversor_error(),
                    max_grad[0],
                    max_grad[1]
                )
            )
        if epoch % plot_time == 0:
            with open('test_GAN_g.pkl', 'wb') as f:
                pickle.dump(generator.generator, f)
            with open('test_GAN_r.pkl', 'wb') as f:
                pickle.dump(generator.reversor, f)
            with open('test_GAN_d.pkl', 'wb') as f:
                pickle.dump(discriminator, f)
            example_graph()

    end_time = timeit.default_timer()
    print(('The code ran for %.2fs' % (end_time - start_time)))

def example_graph(length=20):
    import matplotlib.pyplot as plt
    generator = pickle.load(open('test_GAN_g.pkl','rb'))
    predict_model = theano.function(
        inputs=[generator.input],
        outputs=generator.output)
    data_input = np.reshape(np.linspace(-1.0,1.0,length),(length,1))
    predicted_values = predict_model(data_input)
    plt.scatter(predicted_values[:,0],predicted_values[:,1])
    plt.show()

if __name__ == "__main__":
    example_train([sample_data(10000),sample_data(10000)])
        
