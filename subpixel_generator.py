from __future__ import print_function,division
import numpy as np
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from ops import tmax,tmin,tbox
from optimize import rmsprop,sgd
from theano.tensor.shared_randomstreams import RandomStreams
from lmlp import LMLP

class Subpixel_Layer(object):
    def __init__(self,rng,input,shape,W=None,b=None,init=2):
        #shape =(num images(0),
        #        height(1),width(2),
        #        filter height(3), filter width(4)
        #        multiplier(5),
        #        num input feature maps(6),
        #        num intermediate filters per output map(7), 
        #        num output feature maps(8),

        #correspondes to "valid" filter

        filter_shape = [shape[7],shape[5]*shape[5]*shape[8],shape[6],shape[3],shape[4]]
        image_shape  = [shape[0],shape[6],shape[1],shape[2]]
        bias_shape   = [shape[7],shape[5]*shape[5]*shape[8]]
        output_shape = [shape[0],shape[8],shape[5]*(shape[1]-shape[3]+1),shape[5]*(shape[2]-shape[4]+1)]

        self.input=input 
        self.shape=shape
        n_in = shape[3]*shape[4]*shape[6]
        self.n_params=np.prod(filter_shape)+np.prod(bias_shape)
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
        for i in range(shape[7]):
            conv_out=conv2d(
                input=input,
                filters=self.W[i],
                border_mode='valid',
                filter_shape=filter_shape[1:],
                input_shape=image_shape
            )
            intermediate.append(conv_out+self.b[i].dimshuffle('x',0,'x','x'))
        self.params=[self.W,self.b]
        self.pre_output=T.max(intermediate,axis=0)
        #reshape the output
        self.output=T.zeros(output_shape)
        r = shape[5]
        for x in range(r): 
            for y in range(r):
                self.output=T.set_subtensor(
                    self.output[:,:,x::r,y::r],self.pre_output[:,r*x+y::r*r,:,:])
        #now it has shape [shape[0],shape[8]*shape[5]*shape[5],shape[1]-shape[3]+1,shape[2]-shape[4]+1]
        self.pre_gradient_norms=T.sum(abs(self.W),axis=(2,3,4))
        self.gradient_norms=tmax(self.pre_gradient_norms,1.0)
        self.max_gradient=T.max(self.pre_gradient_norms)


class LGNN(object):
    def __init__(self,rng,input,shape_layers,params=None,init=2):
        self.input=input
        current_input=self.input
        self.layers=[]
        self.max_gradient=1.0
        self.n_params=0
        if params is None:
            for shape in shape_layers:
                self.layers.append(Subpixel_Layer(
                    rng=rng,
                    input=current_input,
                    shape=shape,
                    init=init
                ))
                current_input=self.layers[-1].output
                self.max_gradient*=self.layers[-1].max_gradient
                self.n_params+=self.layers[-1].n_params
        else:
            index = 0
            for shape in shape_layers:
                self.layers.append(Subpixel_Layer(
                    rng=rng,
                    input=current_input,
                    shape=shape,
                    W=params[index],
                    b=params[index+1]
                ))
                index+=2
                current_input=self.layers[-1].output
                self.max_gradient*=self.layers[-1].max_gradient
                self.n_params+=self.layers[-1].n_params
                
        self.output=self.layers[-1].output
        self.params = [param for layer in self.layers for param in layer.params]

def example_train(n_epochs=1000):
    batch_size=10
    from load_mnist import load_data_mnist
    import timeit
    from PIL import Image
    
    GNN_shape=[[batch_size, 6, 6,3,3,2,512,2,256],
               [batch_size, 8, 8,3,3,2,256,2,128],
               [batch_size,12,12,3,3,2,128,2, 64],
               [batch_size,20,20,3,3,2, 64,2, 32],
               [batch_size,36,36,5,5,1, 32,2, 16],
               [batch_size,32,32,5,5,1, 16,2,  1]]
    
    datasets = load_data_mnist()
    train_set_x, train_set_y = datasets[0]

    print('... building the model')

    index = T.lscalar() 
    z = T.matrix('z') 
    w = T.matrix('w')  
    rng = np.random.RandomState(1001)
    fc_layer=LMLP(
        rng,
        input=z,
        info_layers=[[2,batch_size,6*6*512]],
        init=1
    )
    generator_input = fc_layer.output.reshape((batch_size,512,6,6))
    generator = LGNN(
        rng,
        input=generator_input,
        shape_layers=GNN_shape,
        init=1
    )
    output=generator.output
    box_output=tbox(output,0.0,1.0)
    print('number of parameters: ' + str(fc_layer.n_params+generator.n_params))
    params = fc_layer.params+generator.params
    max_gradient = fc_layer.max_gradient*generator.max_gradient
    get_gradient_max = theano.function(
        inputs=[],
        outputs=max_gradient,
        givens={
        }
    )
    training_input=np.zeros((batch_size,batch_size))
    training_input[(range(batch_size),range(batch_size))]=1.0
    training_input=theano.shared(np.asarray(training_input,
                                            dtype=theano.config.floatX),
                                 borrow=True)
    truth=w.reshape((batch_size,1,28,28))
    cost = T.mean((output-truth)**2)
    updates=sgd(cost,params,0.001)
    train_model = theano.function(
        inputs=[],
        outputs=cost,
        updates=updates,
        givens={
            z: training_input,
            w: train_set_x[:batch_size]
        }
    )
    export_model = theano.function(
        inputs=[],
        outputs=box_output,
        givens={
            z: training_input
        }
    )
    start_time = timeit.default_timer()
    for epoch in range(n_epochs):
        if epoch%100 == 1:
            images=255*export_model()
            for i in range(batch_size):
                array=np.array(images[i])
                print(array.shape)
                array=array.reshape((28,28))
                im=Image.fromarray(array).convert('L')
                im.save('mnist_'+str(i)+'.png')
            print('saved')
        print(train_model())

    end_time = timeit.default_timer()
    print(('The code ran for %.2fs' % (end_time - start_time)))

if __name__ == "__main__":
    example_train()
