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
                 numFilters=[512, 128], 
                 numFactors=[256, 64],
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
            assert len(load_layers) == depth, \
                "loading layers count must equal depth of pyramid"
        
        X = X.astype('float32')  # hopefully we don't run OOMem here..
        
        if self.normalize_data:
            X -= X.mean(0)[None, :]
            X /= X.std(0)[None, :] + X.std() * 0.1

        input_nbr = depth + 2
        splitter = ImageSplitter(X, n=input_nbr)

        F = self.F
        H = self.H
        lr = learningRate
        dim = splitter.get_dimension()
        numpy_rng = np.random.RandomState(1)
        
        Dim = [dim]
        for i in range(1, depth):
            Dim.append(H[i-1])
        
        U = [None] * depth
        V = [None] * depth
        W = [None] * depth
        b_U = [None] * depth
        b_V = [None] * depth
        b_W = [None] * depth
        
        #M = [None] * int((depth * (depth + 1)) / 2)  # as the pyramid.. 1 + 2 + ...
        M = []
        for layer in range(1, depth + 1):
            elems_per_layer = depth - layer + 1
            M.append([None] * elems_per_layer)
        assert len(M[-1]) == 1, \
            'the last layer of the pyramid most contain only 1 element'
    
        # true batchsize is a combination of batchsize (aka: number of videos) and
        # the ngram (number of frames per training set) and the total number of
        # frames in the respective video
        true_batchsize = splitter.get_batch_size()
        
        
        # inputs
        
        #x = tf.placeholder(tf.float32, [true_batchsize, input_nbr, dim])
        x = tf.placeholder(tf.float32, [None, input_nbr, dim])
        
        for layer in range(0, depth):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # T R A I N  L A Y E R
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if print_debug:
                print("[CONSTRUCT LAYER " + str(layer + 1) + "]")
            
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
                U[layer] = tf.Variable(
                    tf.random_normal(shape=(dim, F[layer])) * 0.01)
                V[layer] = tf.Variable(
                    tf.random_normal(shape=(dim, F[layer])) * 0.01)
                W[layer] = tf.Variable(
                    numpy_rng.uniform(low=-0.01, high=+0.01, 
                                      size=( F[layer],
                                      H[layer])).astype('float32'))
                b_U[layer] = tf.Variable(np.zeros(F[layer], dtype='float32'))
                b_V[layer] = tf.Variable(np.zeros(F[layer], dtype='float32'))
                b_W[layer] = tf.Variable(np.zeros(H[layer], dtype='float32'))
                if print_debug:
                    print("\tcould not preload weights for Layer " +\
                          str(layer + 1))
            
            # m = sigmoid ( W . Ux1 * Vx2 )
            num_hidden_nodes = depth - layer
            for i in range(num_hidden_nodes):
                if layer == 0:
                    _x = tf.squeeze(tf.slice(x, [0, i, 0], [-1, 1, -1]))
                    _y = tf.squeeze(tf.slice(x, [0, i+1, 0], [-1, 1, -1]))
                else:
                    _x = M[layer-1][i]
                    _y =  M[layer-1][i+1]
                m = tf.sigmoid(tf.matmul(tf.multiply(
                    tf.matmul(_x ,U[layer]) + b_U[layer],
                    tf.matmul(_y,V[layer]) + b_V[layer]), W[layer]) + b_W[layer])
                M[layer][i] = m
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # C O S T  F U N C T I O N
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TODO: sofar, this part ONLY works for 2-layer networks!
        
        dim = Dim[0]  # the dimension of the input image
        _x = tf.squeeze(tf.slice(x, [0, input_nbr-2, 0], [-1, 1, -1]))
        m1 = M[-2][-1]
        m2 = M[-1][0]  # the last pyramid layer M has 1 element (always)
        U2m1 = tf.matmul(m1 ,U[-1]) 
        W2_T_m2 = tf.matmul(m2, tf.transpose(W[-1]))
        
        m1_hat = tf.matmul(
            tf.multiply(U2m1, W2_T_m2),
            tf.transpose(V[-1]))
        
        U1_x = tf.matmul(_x, U[-2])
        W1_T_m_hat = tf.matmul(m1_hat, tf.transpose(W[-2]))
        
        # ---
        oz = tf.matmul(tf.multiply(U1_x, W1_T_m_hat), 
                      tf.transpose(V[-2]))
        
        z = tf.squeeze(tf.slice(x, [0, input_nbr-1, 0], [-1, 1, -1]))
        
        cost = tf.nn.l2_loss(oz-z)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)\
            .minimize(cost)
        
        # ---
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # R U N  O P T I M I Z E R
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            test_set = splitter.get_test(ngram=4)
            n = test_set.shape[0]
            
            for epoch in range(epochs):
                
                while splitter.is_same_batch_run():
                    batch = splitter.next_batch(ngram=4)
                    sess.run(optimizer, feed_dict={x: batch})
                
                cost_ = sess.run(cost, feed_dict={x: test_set}) / n 
                if print_debug:
                    print ("Training: Epoch: %03d/%03d cost: %.9f" %\
                               (epoch,epochs ,cost_) )
            
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # store weights as numpy arrays into object
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for layer in range(0, depth):
                self.U_np[layer] = np.array(U[layer].eval(sess))
                self.V_np[layer] = np.array(V[layer].eval(sess))
                self.W_np[layer] = np.array(W[layer].eval(sess))
                self.b_U_np[layer] = np.array(b_U[layer].eval(sess))
                self.b_V_np[layer] = np.array(b_V[layer].eval(sess))
                self.b_W_np[layer] = np.array(b_W[layer].eval(sess))
                
        
