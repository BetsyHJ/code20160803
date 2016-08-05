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

#from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys, re, numpy as np
import timeit, time

import numpy

import theano
import theano.tensor as T

from read_data import *
from readproduct import *
from logistic_sgd import LogisticRegression, load_data


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
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, params):
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
        # print(params[0*2+1].shape) 
        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden[0],
            W = theano.shared(value=params[0*2], name='W', borrow=True),
            b = theano.shared(value=params[0*2+1], name='b', borrow=True),
            activation=T.tanh
        )
	self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer1.output,
            n_in=n_hidden[0],
            n_out=n_hidden[1],
            W = theano.shared(value=params[1*2], name='W', borrow=True),
            b = theano.shared(value=params[1*2+1], name='b', borrow=True),
            activation=T.tanh
        )
	self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer2.output,
            n_in=n_hidden[1],
            n_out=n_hidden[2],
            W = theano.shared(value=params[2*2], name='W', borrow=True),
            b = theano.shared(value=params[2*2+1], name='b', borrow=True),
            activation=T.tanh
        )


        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden[2],
            n_out=n_out,
            W = theano.shared(value=params[3*2], name='W', borrow=True),
            b = theano.shared(value=params[3*2+1], name='b', borrow=True),
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        self.Y_pred = self.logRegressionLayer.Y_pred

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input
def get_nerual_num(line):
    str_temp = re.split(" |\r|\t|\n", line)
    nerual_num = []
    for i in str_temp:
        if i == '':
            continue
        nerual_num.append(int(i))
    return nerual_num
def read_param(params_file):
    #paramter define
    params = []
    #read data
    f = open(params_file)
    nerual_num = get_nerual_num(f.readline())  
    for j in range(len(nerual_num)-1):
        #read W
        line = f.readline()
        if not line:
            #print "read data u error!"
            break
        temp_data = []
        temp_str = re.split(" |\t|\n", line)
        for i in temp_str:
            if i == "":
                continue
    	    temp_data.append(float(i))
        temp_data = np.array(temp_data).reshape(nerual_num[j], nerual_num[j+1])
	#temp_data = theano.shared(np.array(temp_data, dtype = np.float32))
        params.append(temp_data)
        #read B
        line = f.readline()
        if not line:
            #print "read data u error!"
            break
        temp_data = []
        temp_str = re.split(" |\t|\n", line)
        for i in temp_str:
            if i == "":
                continue
            temp_data.append(float(i))
	#temp_data = theano.shared(np.array(temp_data, dtype = np.float32))
        params.append(np.array(temp_data, dtype = np.float32))

    print("reading is finished~")
    f.close()
    return params, nerual_num
    
def test_itemPre(test_user, test_nega, cate_users, user_pro, y_pred):
    number = 0
    P_10 = 0
    R_50 = 0
    MRR = 0
    MAP = 0
    start = time.time()
    for i in test_user: #read u_test firstly, so we can use i to find test user
        pro_count = {}
        if i not in test_nega:
	    print i
            continue
        number = number + 1
        pos_pro = user_pro[i]
        #fp.write(str(pos_pro)+"\n")
        #get topk user from super users
        selected_user = cate_users[y_pred[number-1]]
        #get topk user productlist and count
        for j in selected_user:
            t_user = j
    	    if t_user not in user_pro:
    	        continue
            temp_pro = user_pro[t_user]
            for k in temp_pro:
                if k in pro_count:
                    temp = pro_count[k] + 1
                    pro_count.update({k: temp})
                else:
                    pro_count.setdefault(k, 1)
	#totle_product_pop = {}
        candidate_items = test_nega[i]
        totle_product = {}
        for j in candidate_items:
            if j in pro_count:
                totle_product.setdefault(j, pro_count[j])
            else:
                totle_product.setdefault(j, 0)
        # rank the product
        totle_product = sorted(totle_product.iteritems(), key=lambda d:d[1], reverse = True)
        total_product = []
        t = {}
        for (j, t_count) in totle_product:
            total_product.append(j)
        t = sorted(t.iteritems(), key=lambda d:d[1], reverse = True)
        for (j, _) in t:
            total_product.append(j)
        #fp.write(str(pos_pro)+"\n")
        # get result P@10, R@50, MAP, MRR, AUC
        count = 0
        right_num = 0
        temp_P_10 = 0
        temp_R_50 = 0
        temp_MRR = 0
        temp_MAP = 0
        num = [10, 10]
        for j in total_product:
	    #fp.write(str(j)+" ")
	    if count == num[0]:
	        temp_P_10 = 1.0 * right_num / num[0]
	    if count == num[1]:
	        temp_R_50 = 1.0 * right_num / len(pos_pro)
	    count = count + 1
 	    for k in pos_pro:
	        if k == j:
		    right_num = right_num + 1
		    temp_MAP = temp_MAP + 1.0 * right_num / count
		    if right_num is not 1:
		        break
	        if right_num == 1:
		    temp_MRR = 1.0 / count
		    break
        P_10 = P_10 + temp_P_10
        R_50 = R_50 + temp_R_50
        MRR = MRR + temp_MRR
        MAP = MAP + temp_MAP / right_num
        #fp.write("\n")
        #fp.write(str(temp_P_10)+" "+str(temp_R_50)+" "+str((temp_P_10*10))+"\n")
        if number % 1000 == 0 :
	    print "count : %d, time : %f map : %f" % (number, time.time() - start, temp_MAP)

    P_10 = P_10 / number
    R_50 = R_50 / number
    MRR = MRR / number
    MAP = MAP / number
    print number, P_10, R_50, MAP, MRR
def read_data(filename, test_user):
    f = open(filename)
    user_pros = {}
    while 1:
        line = f.readline()
        if not line:
            break
        str_temp = re.split(" |\t|\n|\r", line)
        user_id = str_temp[0]
        pros = []
        #if user_id not in name:
        #    name.append(user_id)
        if user_id not in test_user:
            continue
        for i in range(1, len(str_temp)):
            if str_temp[i] == "":
                continue
            else:
                pros.append(int(str_temp[i]))
        user_pros[user_id] = pros
    f.close()
    return user_pros

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
    datasets, user_ids = load_data("test_feature1")

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size + 1
    
    params, nerual_num = read_param(sys.argv[1])
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=nerual_num[0],
        n_hidden=nerual_num[1:-1],
        n_out=nerual_num[-1],
        params = params
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.Y_pred(),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        },
	on_unused_input='ignore'
    )
    ###############
    #  TEST MODEL #
    ###############
    print('... testing')

    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    # test it on the test set
    y_pred = []
    for i in range(n_test_batches):
        y_pred += list(test_model(i))
    #y_pred = [test_model(i) for i in range(n_test_batches)]
    print len(y_pred)
    ##test
    wb_Embedfile = "test_feature"#"../jd_wfeature"
    jd_wb_linkfile = "network"#"../../net_work10w"
    test_user =user_ids# get_user(wb_Embedfile, 0)
    user_pro = read_data("../../user_product_list10w", test_user)
    print len(user_pro)
    test_nega = readpro_nega("test_nega.txt")
    cate_users = get_cate_users(jd_wb_linkfile)
    test_itemPre(test_user, test_nega, cate_users, user_pro, y_pred)


if __name__ == '__main__':
    test_mlp()



