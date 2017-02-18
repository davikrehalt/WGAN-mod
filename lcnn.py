from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from lmlp import LMLP
from optimize import rmsprop
from ops import tmax,tmin

class LipConvLayer(object):
    def __init__(self,rng,input,shape,W=None,b=None,init=0):
        #shape =(num images(0),
        #        height(1),width(2),
        #        filter height(3), filter width(4)
        #        num input feature maps(5),
        #        num intermediate filters per output map(6), 
        #        num output feature maps(7),

        #correspondes to "valid" filter

        filter_shape = [shape[6],shape[7],shape[5],shape[3],shape[4]]
        image_shape  = [shape[0],shape[5],shape[1],shape[2]]
        bias_shape   = [shape[6],shape[7]]
        output_shape = [shape[0],shape[7],shape[1]-shape[3]+1,shape[2]-shape[4]+1]

        self.input=input 
        n_in = shape[3]*shape[4]*shape[5]
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
            self.W = theano.shared(
                np.asarray(
                    rng.uniform(low=min_value, high=max_value, size=filter_shape),
                    dtype=theano.config.floatX
                ), borrow=True)
        else:
            self.W=W
        if b is None:
            self.b = theano.shared(
                np.asarray(
                    rng.uniform(low=-0.5, high=0.5, size=bias_shape),
                    dtype=theano.config.floatX
                ), borrow=True)
        else:
            self.b=b

        intermediate=[]
        for i in range(shape[6]):
            conv_out=conv2d(
                input=input,
                filters=self.W[i],
                border_mode='valid',
                filter_shape=filter_shape[1:],
                input_shape=image_shape
            )
            intermediate.append(conv_out+self.b[i].dimshuffle('x',0,'x','x'))
        self.output=T.max(intermediate,axis=0)
        self.params=[self.W,self.b]
        self.pre_gradient_norms=T.sum(abs(self.W),axis=(2,3,4))
        self.gradient_norms=tmax(self.pre_gradient_norms,1.0)
        self.max_gradient=T.max(self.pre_gradient_norms)
        self.gradient_cost=T.sum(1.0/(2.0-self.gradient_norms)-1.0)

class LCNN(object):
    def __init__(self, rng, input, shape_layers,params=None,init=0):
        self.input=input
        current_input=self.input
        self.layers=[]
        self.max_gradient=1.0
        self.gradient_cost=0.0
        if params is None:
            for shape in shape_layers:
                self.layers.append(LipConvLayer(
                    rng=rng,
                    input=current_input,
                    shape=shape,
                    init=init
                ))
                current_input=self.layers[-1].output
                self.max_gradient*=self.layers[-1].max_gradient
                self.gradient_cost+=self.layers[-1].gradient_cost
        else:
            index = 0
            for shape in shape_layers:
                self.layers.append(LipConvLayer(
                    rng=rng,
                    input=current_input,
                    shape=shape,
                    W=params[index],
                    b=params[index+1]
                ))
                index+=2
                current_input=self.layers[-1].output
                self.max_gradient*=self.layers[-1].max_gradient
                self.gradient_cost+=self.layers[-1].gradient_cost
                
        self.output=self.layers[-1].output
        self.params = [param for layer in self.layers for param in layer.params]

def test_mnist(n_epoch=1000,batch_size=40):
    from load_mnist import load_data_mnist
    import timeit
    
    plot_time=100
    valid_time=1
    #one entry per layer
    CNN_shape=[[batch_size,28,28,5,5,1,5,20],[batch_size,24,24,5,5,20,5,50]]
    fc_info=[[5,20*20*50,500],[5,500,10]]

    datasets = load_data_mnist()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    #n_train_batches = 1
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]//batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]//batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]//batch_size

    print('... building the model')

    index = T.lscalar() 
    x = T.matrix('x') 
    y = T.matrix('y')  
    rng = np.random.RandomState(1001)
    reshaped_input = x.reshape((batch_size, 1, 28, 28))
    convnet = LCNN(
        rng,
        input=reshaped_input,
        shape_layers=CNN_shape
    )
    fc_layer_input=convnet.output.flatten(2)
    fc_layer=LMLP(
        rng,
        input=fc_layer_input,
        info_layers=fc_info
    )
    max_gradient = convnet.max_gradient*fc_layer.max_gradient
    params = convnet.params+fc_layer.params
    gradient_cost=fc_layer.gradient_cost+convnet.gradient_cost
    cost = fc_layer.mse(y)+gradient_cost
    updates=rmsprop(cost,params)
    validate_model = theano.function(
        inputs=[index],
        outputs=fc_layer.mse(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    get_gradient_max = theano.function(
        inputs=[],
        outputs=max_gradient,
        givens={
        }
    )
    print('... training')

    start_time = timeit.default_timer()

    epoch = 0

    for epoch in range(n_epoch):
        if epoch % valid_time == 0:
            validation_losses = [validate_model(i) for i
                                    in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)

            print(
                'epoch %i, error %f, gradient max %f' %
                (
                    epoch,
                    this_validation_loss,
                    get_gradient_max()
                )
            )
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)

    end_time = timeit.default_timer()
    print(('The code ran for %.2fs' % (end_time - start_time)))

if __name__ == "__main__":
    test_mnist()
