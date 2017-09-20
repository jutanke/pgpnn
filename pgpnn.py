import numpy as np

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