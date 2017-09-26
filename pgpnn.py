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
        
        self.frames_per_video = N
        self.vector_dimension = H * W
        self.same_batch_run = True
        
    def get_dimension(self):
        """ gets the data dimension
        """
        return self.vector_dimension
   
    def get_batch_size(self, ngram=None):
        """ calculates the true batchsize
        """
        if ngram is None:
            n = self.n
        else:
            n = ngram
        
        M = self.frames_per_video
        seqs = M - n + 1
        bs = self.batchsize
        return seqs * bs
        
        

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
                 numFilters=[512, 512], 
                 numFactors=[256, 256],
                 modelname=None,
                 normalize_data=True):
        assert depth > 0
        assert depth == 2, "Other depth than 2 not supported"
        assert len(numFactors) == depth, "Number of filters must equal depth"
        assert len(numFilters) == depth, "Number of hidden units must equal depth"
        self.depth = depth
        self.F = numFactors
        self.H = numFilters
        self.is_trained = False
        self.normalize_data = normalize_data
        self.modelname = modelname
        self.is_first_layer_trained = False
        self.is_second_layer_trained = False
        
        self.U_np = [None] * depth
        self.V_np = [None] * depth
        self.W_np = [None] * depth
        self.b_U_np = [None] * depth
        self.b_V_np = [None] * depth
        self.b_W_np = [None] * depth

    def load_stage(self, stage_level):
        """ Tries to load the stage from file
        
        returns True if stage was successfully loaded, otherwise: False
        """
        modelname = self.modelname
        assert self.modelname is not None
        try:
            self.U_np[stage_level] = np.load(modelname + "U" + str(stage_level) + ".npy")
            self.V_np[stage_level] = np.load(modelname + "V" + str(stage_level) + ".npy")
            self.W_np[stage_level] = np.load(modelname + "W" + str(stage_level) + ".npy")
            self.b_U_np[stage_level] = np.load(modelname + "b_U" + str(stage_level) + ".npy")
            self.b_V_np[stage_level] = np.load(modelname + "b_V" + str(stage_level) + ".npy")
            self.b_W_np[stage_level] = np.load(modelname + "b_W" + str(stage_level) + ".npy")
            return True
        except FileNotFoundError:
            return False
    
    def save_stage(self, stage_level):
        """ Saves the stage to file
        """
        modelname = self.modelname
        assert(self.is_trained)
        assert modelname is not None
        np.save(modelname + "U" + str(stage_level), self.U_np[stage_level])
        np.save(modelname + "V" + str(stage_level), self.V_np[stage_level])
        np.save(modelname + "W" + str(stage_level), self.W_np[stage_level])
        np.save(modelname + "b_U" + str(stage_level), self.b_U_np[stage_level])
        np.save(modelname + "b_V" + str(stage_level), self.b_V_np[stage_level])
        np.save(modelname + "b_W" + str(stage_level), self.b_W_np[stage_level])
    
    
    def train(self, X, epochs=100, pre_epochs=100, print_debug=True,
             load_layers=None,
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
        depth = self.depth
        
        
        if load_layers is None:
            load_layers = [False] * depth
        else:
            assert len(load_layers) == depth, "loading layers count must equal depth of pyramid"
        
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
        
        Dim = [dim]
        for i in range(1, depth):
            Dim.append(H[i-1])
        print("DIM", Dim)
        
        U = [None] * depth
        V = [None] * depth
        W = [None] * depth
        b_U = [None] * depth
        b_V = [None] * depth
        b_W = [None] * depth
        
        #M = [None] * int((depth * (depth + 1)) / 2)  # as the pyramid.. 1 + 2 + ...
        M = []
        for layer in range(1, depth + 1):
            elems_per_layer = depth - layer + 2
            M.append([None] * elems_per_layer)
    
        # true batchsize is a combination of batchsize (aka: number of videos) and
        # the ngram (number of frames per training set) and the total number of
        # frames in the respective video
        true_batchsize = splitter.get_batch_size()
        
        
        # inputs
        x = tf.placeholder(tf.float32, [true_batchsize, dim, depth + 2])
        
        
        for layer in range(0, depth):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # T R A I N  L A Y E R
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if print_debug:
                print("[TRAIN LAYER " + str(layer + 1) + "]")
            
            weights_are_pre_loaded = False
            if load_layers[layer]:
                # we want to pre-load the layer
                weights_are_pre_loaded = self.load_stage(layer + 1) 
            
            dim = Dim[layer]
            
            if weights_are_pre_loaded:
                U[layer] = tf.Variable(self.U_np[layer])
                V[layer] = tf.Variable(self.V_np[layer])
                W[layer] = tf.Variable(self.W_np[layer])
                b_U[layer] = tf.Variable(self.b_U_np[layer])
                b_V[layer] = tf.Variable(self.b_V_np[layer])
                b_W[layer] = tf.Variable(self.b_W_np[layer])
                if print_debug:
                    print("\tpre-loading weights for Layer " + str(layer + 1))
            else:
                # randomly initialize the weights
                U[layer] = tf.Variable(tf.random_normal(shape=(dim, F[layer])) * 0.01)
                V[layer] = tf.Variable(tf.random_normal(shape=(dim, F[layer])) * 0.01)
                W[layer] = tf.Variable(
                    numpy_rng.uniform(low=-0.01, high=+0.01, 
                                      size=( F[layer], H[layer])).astype('float32'))
                b_U[layer] = tf.Variable(np.zeros(F[layer], dtype='float32'))
                b_V[layer] = tf.Variable(np.zeros(F[layer], dtype='float32'))
                b_W[layer] = tf.Variable(np.zeros(H[layer], dtype='float32'))
                if print_debug:
                    print("\tcould not preload weights for Layer " + str(layer + 1))
            
            # m = sigmoid ( W . Ux1 * Vx2 )
            num_hidden_nodes = depth - layer + 1
            for i in range(num_hidden_nodes):
                if layer == 0:
                    _x = tf.squeeze(tf.slice(x, [0, 0, i], [-1, dim, 1]))
                    _y = tf.squeeze(tf.slice(x, [0, 0, i+1], [-1, dim, 1]))
                else:
                    _x = M[layer-1][i]
                    _y =  M[layer-1][i+1]
                m = tf.sigmoid(tf.matmul(tf.multiply(
                    tf.matmul(_x ,U[layer]) + b_U[layer],
                    tf.matmul(_y,V[layer]) + b_V[layer]), W[layer]) + b_W[layer])
                M[layer][i] = m
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            test_set = splitter.get_test(ngram=3)
            #X_ = test_set[:,0,:]
            #Y_ = test_set[:,1,:]
            #Z_ = test_set[:,2,:]
            #n = test_set.shape[0]
            #for epoch in range(epochs):
            #    while splitter.is_same_batch_run():
            #        batch = splitter.next_batch(ngram=3)
            #        batch_xs = batch[:,0,:]
            #        batch_ys = batch[:,1,:]
            #        batch_zs = batch[:,2,:]
            #        sess.run(optimizer_2, feed_dict={x: batch_xs, y: batch_ys, z: batch_zs})
            #        sess.run(normalize_U2)
            #        sess.run(normalize_V2)

            #    cost_ = sess.run(cost_2, feed_dict={x: X_, y: Y_, z: Z_}) / n
            #    if print_debug:
            #        print ("Training: Epoch: %03d/%03d cost: %.9f" %\
            #                   (epoch,epochs ,cost_) )
        
        
        #x = tf.placeholder(tf.float32, [None, dim])
        #y = tf.placeholder(tf.float32, [None, dim])
        #z = tf.placeholder(tf.float32, [None, dim])
        
        
            
        
        return
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