import numpy as np
import tensorflow as tf

# this function is 'borrowed' from keras
def random_binomial(shape, p=0.0, dtype=None, seed=None):
    """Returns a tensor with random binomial distribution of values.
    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        p: A float, `0. <= p <= 1`, probability of binomial distribution.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.
    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = 'float32'
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.where(tf.random_uniform(shape, dtype=dtype, seed=seed) <= p,
                    tf.ones(shape, dtype=dtype),
                    tf.zeros(shape, dtype=dtype))

class ImageSplitter:
    """ splits the video set into shorts of n frames
    """
    def __init__(self, 
                 dataset,
                 n=3, 
                 ntest=256,
                 batchsize=64):
        """ ctor
            dataset: list of videos that have to be cut into n-grams
            n: integer, length of subsequence
            ntest: integer, number of items that belong to the test set
            batchsize: ~
            
        The data has to be arranged in following order:
            NUMBER_OF_VIDEOS, LENGTH_OF_VIDEO, HEIGHT, WIDTH
            
        """
        
        self.batchsize = batchsize
        self.batch_loop = 0
        self.n = n
        

        np.random.shuffle(dataset)
        
        self.test_set = dataset[0:ntest]
        self.train_set = dataset[ntest:]
        
        F, N, H, W = dataset.shape # Frames, Numbers, Height, Width
        
        
        self.vector_dimension = H * W
        self.same_batch_run = True
        
    def get_dimension(self):
        """ gets the data dimension
        """
        return self.vector_dimension

    def transform_to_n_gram(self, batch, ngram=None):
        """ transforms the batch into an n-gram (parameter n)
        batch: np.array((batchsize, 20, 64, 64))
        """
        if ngram is None:
            n = self.n
        else:
            n = ngram
        N, M, H, W = batch.shape
        seqs = M-n+1 # sequences per video
        
        Result = np.zeros((seqs * N, n, H, W))
        
        pos = 0
        for j in range(N):
            for i in range(seqs):
                Result[pos] = batch[j, i:i+n]
                pos += 1
        
        N, M, H, W = Result.shape
        return Result.reshape((N, M, H * W))
    
    def is_same_batch_run(self):
        if self.same_batch_run:
            return True
        else:
            self.same_batch_run = True
            return False
        
    
    def next_batch(self, ngram=None):
        """ returns the next batch
        """
        start = self.batch_loop
        end = self.batch_loop + self.batchsize
        N = self.train_set.shape[0]
        if N > end:
            self.same_batch_run = True
            self.batch_loop = end
            return self.transform_to_n_gram(
                self.train_set[start:end], ngram=ngram)
        else:
            self.same_batch_run = False
            diff = (N - start)
            end = self.batchsize - diff
            
            set1 = self.train_set[start:N]
            set2 = self.train_set[0:end]
            
            self.batch_loop = end
            return self.transform_to_n_gram(
                np.concatenate((set1, set2)), ngram=ngram)
    
    def get_train(self, ngram=None):
        """ returns the train data as ngram
        """
        return self.transform_to_n_gram(self.train_set, ngram=ngram)
    
    def get_test(self, ngram=None):
        """ returns the test data as ngram
        """
        return self.transform_to_n_gram(self.test_set, ngram=ngram)

    
class PredictiveGatingPyramid:
    """ implementation of the pgp
    """
    
    def __init__(self, 
                 depth=2, 
                 numfilters=512, 
                 numHidden=256,
                 modelname=None,
                 normalize_data=True):
        assert depth > 0
        assert depth == 2, "Other depth than 2 not supported"
        self.depth = depth
        self.F = numfilters
        self.H = numHidden
        self.is_trained = False
        self.normalize_data = normalize_data
        self.modelname = modelname
        self.is_first_layer_trained = False
        self.is_second_layer_trained = False
        
        
    def save_first_stage(self):
        """ saves the model onto disk
        """
        modelname = self.modelname
        assert(self.is_trained)
        assert modelname is not None
        np.save(modelname + "U1", self.U1_np)
        np.save(modelname + "V1", self.V1_np)
        np.save(modelname + "W1", self.W1_np)
        np.save(modelname + "b_U1", self.b_U1_np)
        np.save(modelname + "b_V1", self.b_V1_np)
        np.save(modelname + "b_W1", self.b_W1_np)
        np.save(modelname + "b_U1_out", self.b_U1_out_np)
        np.save(modelname + "b_V1_out", self.b_V1_out_np)
        np.save(modelname + "b_W1_out", self.b_W1_out_np)
    
    def load_first_stage(self, modelname=None):
        """ loads the first layer of the network
        """
        if modelname is None:
            modelname = self.modelname
        self.U1_np = np.load(modelname + "U1.npy")
        self.V1_np = np.load(modelname + "V1.npy")
        self.W1_np = np.load(modelname + "W1.npy")
        self.b_U1_np = np.load(modelname + "b_U1.npy")
        self.b_V1_np = np.load(modelname + "b_V1.npy")
        self.b_W1_np = np.load(modelname + "b_W1.npy")
        self.b_U1_out_np = np.load(modelname + "b_U1_out.npy")
        self.b_V1_out_np = np.load(modelname + "b_V1_out.npy")
        self.b_W1_out_np = np.load(modelname + "b_W1_out.npy")
        self.is_first_layer_trained = True
    
    def save_second_stage(self):
        """ saves the model onto disk
        """
        modelname = self.modelname
        assert(self.is_trained)
        assert modelname is not None
        np.save(modelname + "U2", self.U2_np)
        np.save(modelname + "V2", self.V2_np)
        np.save(modelname + "W2", self.W2_np)
        np.save(modelname + "b_U2", self.b_U2_np)
        np.save(modelname + "b_V2", self.b_V2_np)
        np.save(modelname + "b_W2", self.b_W2_np)
    
    def load_second_stage(self, modelname=None):
        """ load second layer of the network
        
            returns True if the layers were successfully loaded,
                otherwise False
        """
        if modelname is None:
            modelname = self.modelname
        assert modelname is not None
        try:
            self.U2_np = np.load(modelname + "U2.npy")
            self.V2_np = np.load(modelname + "V2.npy")
            self.W2_np = np.load(modelname + "W2.npy")
            self.b_U2_np = np.load(modelname + "b_U2.npy")
            self.b_V2_np = np.load(modelname + "b_V2.npy")
            self.b_W2_np = np.load(modelname + "b_W2.npy")
            self.is_second_layer_trained = True
            return True
        except FileNotFoundError:
            self.is_second_layer_trained = False
            return False
    
    
    def train(self, X, epochs=100, pre_epochs=100, print_debug=True,
             load_first_stage=False, force_pretrain_first_stage=False,
             load_second_stage=True, force_train_second_stage=True,
             learningRate=0.0001, save_results=True):
        """ trains the model
        
            X: training data: must be organized as follows:
                Number_of_videos, video_length, H, W
            epochs: number of epochs to run for final training
            pre_epochs: number of epochs for training initial layer
            load_first_stage: if True, then the first stage will not
                be trained separatly but will be loaded from file
                instead
        """
        self.is_trained = True
        
        X = X.astype('float32')  # hopefully we don't run OOMem here..
        
        if self.normalize_data:
            X -= X.mean(0)[None, :]
            X /= X.std(0)[None, :] + X.std() * 0.1

        splitter = ImageSplitter(X, n=self.depth+1)

        F = self.F
        H = self.H
        lr = learningRate
        dim = splitter.get_dimension()
        numpy_rng = np.random.RandomState(1)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # first layer
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # pretrain the early layer for faster convergence
        if load_first_stage:
            self.load_first_stage()
        
        if not self.is_first_layer_trained or force_pretrain_first_stage:
            x = tf.placeholder(tf.float32, [None, dim])
            y = tf.placeholder(tf.float32, [None, dim])
    
            if self.is_first_layer_trained:
                U1 = tf.Variable(self.U1_np)
                V1 = tf.Variable(self.V1_np)
                W1 = tf.Variable(self.W1_np)

                b_U1 = tf.Variable(self.b_U1_np)
                b_V1 = tf.Variable(self.b_V1_np)
                b_W1 = tf.Variable(self.b_W1_np)
                b_U1_out = tf.Variable(self.b_U1_out_np)
                b_V1_out = tf.Variable(self.b_V1_out_np)
                b_W1_out = tf.Variable(self.b_W1_out_np)
            else:
                U1 = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
                V1 = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
                W1 = tf.Variable(
                    numpy_rng.uniform(low=-0.01, high=+0.01, 
                                      size=( F, H)).astype('float32'))

                b_U1 = tf.Variable(np.zeros(F, dtype='float32'))
                b_V1 = tf.Variable(np.zeros(F, dtype='float32'))
                b_W1 = tf.Variable(np.zeros(H, dtype='float32'))
                b_U1_out = tf.Variable(np.zeros(dim, dtype='float32'))
                b_V1_out = tf.Variable(np.zeros(dim, dtype='float32'))
                b_W1_out = tf.Variable(np.zeros(F, dtype='float32'))

            m1 = tf.sigmoid(tf.matmul(tf.multiply(
                tf.matmul(x,U1) + b_U1,
                tf.matmul(y,V1) + b_V1), W1) + b_W1)

            ox = tf.matmul(tf.multiply(
                    tf.matmul(m1,tf.transpose(W1)) + b_W1_out,
                    tf.matmul(y,V1) + b_V1),tf.transpose(U1))+ b_U1_out
            oy = tf.matmul(tf.multiply(
                    tf.matmul(m1,tf.transpose(W1)) + b_W1_out,
                    tf.matmul(x,U1) + b_U1), 
                tf.transpose(V1)) + b_V1_out

            cost_1 = tf.nn.l2_loss(ox-x) + tf.nn.l2_loss(oy-y)
            optimizer_1 = tf.train.AdamOptimizer(learning_rate=lr)\
                .minimize(cost_1)

            norm_U1 = tf.nn.l2_normalize(U1, [0,1], epsilon=1e-12, name=None)
            norm_V1 = tf.nn.l2_normalize(V1, [0,1], epsilon=1e-12, name=None)

            normalize_U1 = U1.assign(norm_U1)
            normalize_V1 = V1.assign(norm_V1)

            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                test_set = splitter.get_test(ngram=2)
                X_ = test_set[:,0,:]
                Y_ = test_set[:,1,:]
                n = test_set.shape[0]
                for epoch in range(pre_epochs):
                    while splitter.is_same_batch_run():
                        batch = splitter.next_batch(ngram=2)
                        batch_xs = batch[:,0,:]
                        batch_ys = batch[:,1,:]
                        sess.run(optimizer_1, feed_dict={x: batch_xs, y: batch_ys})
                        sess.run(normalize_U1)
                        sess.run(normalize_V1)

                    cost_ = sess.run(cost_1, feed_dict={x: X_, y: Y_}) / n
                    if print_debug:
                        print ("Pretrain: Epoch: %03d/%03d cost: %.9f" %\
                                   (epoch,pre_epochs ,cost_) )

                self.U1_np = np.array(U1.eval(sess))
                self.V1_np = np.array(V1.eval(sess))
                self.W1_np = np.array(W1.eval(sess))
                self.b_U1_np = np.array(b_U1.eval(sess))
                self.b_V1_np = np.array(b_V1.eval(sess))
                self.b_W1_np = np.array(b_W1.eval(sess))
                self.b_U1_out_np = np.array(b_U1_out.eval(sess))
                self.b_V1_out_np = np.array(b_V1_out.eval(sess))
                self.b_W1_out_np = np.array(b_W1_out.eval(sess))
                self.is_first_layer_trained = True
                
                if self.modelname is not None and save_results:
                    self.save_first_stage()
        
        if print_debug:
            print("pre-training ended")
            
        # start training the second stage
        # set of first stage
        
        x = tf.placeholder(tf.float32, [None, dim])
        y = tf.placeholder(tf.float32, [None, dim])
        z = tf.placeholder(tf.float32, [None, dim])
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # second layer
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if load_second_stage and self.load_second_stage():
            # second stage is loaded, initialize with
            # given weights!
            U2 = tf.Variable(self.U2_np)
            V2 = tf.Variable(self.V2_np)
            W2 = tf.Variable(self.W2_np)

            b_U2 = tf.Variable(self.b_U2_np)
            b_V2 = tf.Variable(self.b_V2_np)
            b_W2 = tf.Variable(self.b_W2_np)
            if print_debug:
                print("pre-load second layer weights")
        else:
            # second stage weights could not be loaded:
            # initialize randomly
            U2 = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
            V2 = tf.Variable(tf.random_normal(shape=(dim, F)) * 0.01)
            W2 = tf.Variable(
                numpy_rng.uniform(low=-0.01, high=+0.01, 
                                  size=( F, H)).astype('float32'))

            b_U2 = tf.Variable(np.zeros(F, dtype='float32'))
            b_V2 = tf.Variable(np.zeros(F, dtype='float32'))
            b_W2 = tf.Variable(np.zeros(H, dtype='float32'))
            if print_debug:
                print("could not pre-load second layer -> randomly initialize")
        
        
        m1 = tf.sigmoid(tf.matmul(tf.multiply(
                tf.matmul(x,U2) + b_U2,
                tf.matmul(y,V2) + b_V2), W2) + b_W2)
        
        #m2 = tf.sigmoid(tf.matmul(tf.multiply(
        #        tf.matmul(y,U1) + b_U1,
        #        tf.matmul(z,V1) + b_V1), W1) + b_W1)
        
        Uy = tf.matmul(y, U2)
        WTm1 = tf.matmul(m1, tf.transpose(W2))
        
        o3 = tf.matmul(tf.multiply(Uy, WTm1), tf.transpose(V2))

        cost_2 = tf.nn.l2_loss(o3-z)
        optimizer_2 = tf.train.AdamOptimizer(learning_rate=lr)\
            .minimize(cost_2)
        
        norm_U2 = tf.nn.l2_normalize(U2, [0,1], epsilon=1e-12, name=None)
        norm_V2 = tf.nn.l2_normalize(V2, [0,1], epsilon=1e-12, name=None)

        normalize_U2 = U2.assign(norm_U2)
        normalize_V2 = V2.assign(norm_V2)
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            test_set = splitter.get_test(ngram=3)
            X_ = test_set[:,0,:]
            Y_ = test_set[:,1,:]
            Z_ = test_set[:,2,:]
            n = test_set.shape[0]
            for epoch in range(epochs):
                while splitter.is_same_batch_run():
                    batch = splitter.next_batch(ngram=3)
                    batch_xs = batch[:,0,:]
                    batch_ys = batch[:,1,:]
                    batch_zs = batch[:,2,:]
                    sess.run(optimizer_2, feed_dict={x: batch_xs, y: batch_ys, z: batch_zs})
                    sess.run(normalize_U2)
                    sess.run(normalize_V2)

                cost_ = sess.run(cost_2, feed_dict={x: X_, y: Y_, z: Z_}) / n
                if print_debug:
                    print ("Training: Epoch: %03d/%03d cost: %.9f" %\
                               (epoch,epochs ,cost_) )
            
            # safe weights locally
            self.U2_np = np.array(U2.eval(sess))
            self.V2_np = np.array(V2.eval(sess))
            self.W2_np = np.array(W2.eval(sess))
            self.b_U2_np = np.array(b_U2.eval(sess))
            self.b_V2_np = np.array(b_V2.eval(sess))
            self.b_W2_np = np.array(b_W2.eval(sess))
            self.is_second_layer_trained = True
        
        
        if self.modelname is not None:
            self.save_second_stage()

        if print_debug:
            print("training is finished")