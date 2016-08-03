"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""


__docformat__ = 'restructedtext en'


import os
import sys
import timeit, time

import numpy, random

import theano
import theano.tensor as T

from read_data import *
#from logistic_sgd import LogisticRegression, load_data

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP_U(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        
        # for u
        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_out[0],
            activation=T.tanh
        )
	self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_out[0],
            n_out=n_out[1],
            activation=T.tanh
        )
	self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer2.output,
            n_in=n_out[1],
            n_out=n_out[2],
            activation=T.tanh
        )
        self.output = self.hiddenLayer.output
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params =self.hiddenLayer1.params+self.hiddenLayer2.params+ self.hiddenLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input
# start-snippet-2
class MLP_E(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_out[0],
            activation=T.tanh
        )
	self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_out[0],
            n_out=n_out[1],
            activation=T.tanh
        )
	self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer2.output,
            n_in=n_out[1],
            n_out=n_out[2],
            activation=T.tanh
        )
        self.params = self.hiddenLayer1.params+self.hiddenLayer2.params+ self.hiddenLayer.params
        self.input = input
        self.output = self.hiddenLayer.output

def create_matrix(num_train, flag):
    temp = np.zeros((num_train, 2*num_train))
    for i in range(num_train):
        temp[i][num_train*flag + i] = 1
    return temp        
def load_data(borrow=True):
    jd_Embfile = "embedding"#"../../network10.embeddings512"
    wb_Embedfile = "train_feature"#"../jd_wfeature"
    jd_wb_linkfile = "network"#"../../net_work10w"
    size_e, e_train = get_vector(jd_Embfile, 1)
    size_u, u_train = get_vector(wb_Embedfile, 0)
    u_c = get_net(jd_wb_linkfile)
    negative_set = 5
    for cross_i in range(1):#5 cross check
 	#params_file = "./result_all/result_WB_cross_512_nega1_init"#+str(cross_i)
        #Print(y, params_file, nerual_num_u, nerual_num_e)	
        #test_user = e_user[-1:]
        #print size#, np.shape(e_train)[1]
        iteration = 1
	e_user = e_train.keys(); e_size = len(e_user)
        for iter in range(iteration): 
            count = 0
	    cost_tt = 0.0
            
            u_input = []
            e1_input = []
            e2_input = []
            for i in u_train:
                if i not in u_c:
                    count = count + 1
                    continue
                count = count + 1
                c_list = u_c[i][:]
                for j1 in range(len(c_list)):
                    c_Nega = []
                    for j2 in range(negative_set):
                        if c_list[j1] not in e_train:
                            continue
                        t = random.randint(0, e_size-1)
                        while (t in c_Nega) or (e_user[t] in c_list) or (e_user[t] == i) :
                            t = random.randint(0, e_size-1)    
                        #print  u_train[user[t]]         
                        #if count == 6 or count ==7 :
                        #print e_train[i]
			c_Nega.append(t)
                        u_input.append(u_train[i])
                        e1_input.append(e_train[c_list[j1]])
                        e2_input.append(e_train[e_user[t]])
                        #train(e_train[i].get_value(), u_train[c_list[j1]].get_value(), u_train[user[t]].get_value())
                c_Nega = []
                if len(u_input)==0 or len(e1_input)==0 or len(e2_input)==0:
                    continue
            print len(u_input[0]), len(e1_input[0]), len(e2_input[0])
            u_input = theano.shared(numpy.asarray(u_input, dtype=theano.config.floatX), borrow=borrow)
            e1_input = theano.shared(numpy.asarray(e1_input, dtype=theano.config.floatX), borrow=borrow)
            e2_input = theano.shared(numpy.asarray(e2_input, dtype=theano.config.floatX), borrow=borrow)
    return u_input, e1_input, e2_input
 
def Print(y, params_file, nerual_num_u, nerual_num_c):
    file = open(params_file, 'w')
    file.write(' '.join(map(lambda x: str(x), nerual_num_u))+"\n")
    file.write(' '.join(map(lambda x: str(x), nerual_num_c))+"\n")
    params = y
    for i in range(len(params)/2):
        temp_data = params[i*2].get_value()
        d1 = temp_data.shape[0]
        d2 = temp_data.shape[1]
         
        data = temp_data.reshape(1, d1 * d2)[0].tolist()
        file.write(' '.join(map(lambda x: str(x), data)))
        file.write("\n")
        file.write(' '.join(map(lambda x: str(x), params[i*2+1].get_value())))
        file.write("\n")
    file.close()
   
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    # load data
    u_input, e1_input, e2_input = load_data()    
    nerual_num_U = [512, 512, 512]
    nerual_num_E = [512, 512, 512]

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    num_train = batch_size
    gamma = 0.2
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    u = T.matrix('u')  # the data is presented as rasterized images
    e = T.matrix('e')

    rng = numpy.random.RandomState(1234)

    # for weibo user
    model1 = MLP_U(
        rng=rng,
        input=u,
        n_in=717,
        n_hidden=n_hidden,
        n_out=nerual_num_U
    )
    # for ecommerce user
    model2 = MLP_E(
        rng=rng,
        input=e,
        n_in=512,
        n_hidden=n_hidden,
        n_out=nerual_num_E
    )
    
    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    yu = model1.output
    ye = model2.output #make the input is [e1_batch, e2_batch], so i get [ye1, ye2]
    t0 = create_matrix(num_train, 0)
    t1 = create_matrix(num_train, 1)
    ye1 = T.dot(t0, ye)
    ye2 = T.dot(t1, ye)
    t = np.ones(nerual_num_U[-1])
    #cost = T.sum(T.log(1+T.exp( -gamma*(T.dot(yu*ye1, t)/(T.sqrt(T.dot(yu*yu, t)*T.dot(ye1*ye1, t)))-T.dot(yu*ye2, t)/(T.sqrt(T.dot(yu*yu, t)*T.dot(ye2*ye2, t)))))))
    cost_pair = T.dot(yu*ye1, t)-T.dot(yu*ye2, t)
    cost = T.sum(T.log(1+T.exp(-gamma*cost_pair)))
    

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    Params = model1.params + model2.params
    gparams = [T.grad(cost, param) for param in Params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(Params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost_pair,
        updates=updates,
        givens={
            u: u_input[index * batch_size: (index + 1) * batch_size],
            e: T.concatenate((e1_input[index * batch_size: (index + 1) * batch_size],e2_input[index * batch_size: (index + 1) * batch_size]))
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    for cross_i in range(1):#5 cross check
 	#params_file = "./result_all/result_WB_cross_512_nega1_init"#+str(cross_i)
        #Print(Params, params_file, nerual_num_U, nerual_num_E)	
        #test_user = e_user[-1:]
        #print size#, np.shape(e_train)[1]
        start = time.time()
        iteration = 1000
	#user = e_user; size = len(user)
        for iter in range(iteration): 
            #right_each = 0
            sum_right = 0
            n_train_batches = u_input.get_value(borrow=True).shape[0] // batch_size
            pair_result = []
            for minibatch_index in range(n_train_batches):
                #print minibatch_index
                if minibatch_index == (n_train_batches - 1):
                    num_train = u_input.get_value(borrow=True).shape[0] % batch_size
                else:
                    num_train = batch_size
                pair_result = train_model(minibatch_index)
                sum_right += len([1 for tt in pair_result if tt > 0])
            if (iter+1) % 5 == 0:
                Print(Params, "./result_all/result_WB_"+"iter"+str(iter+1),[717] + nerual_num_U, [512] + nerual_num_E)
            print "iteration %d -- time: %f" % (iter + 1, (time.time() - start)), " , right:", sum_right, (1.0*sum_right/u_input.get_value(borrow=True).shape[0])
	params_file = "temp"


if __name__ == '__main__':
    test_mlp()





