import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from lyrics import map_lyrics

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell, LSTMCell

# from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs

def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)

def x2sequence(x, T, D, batch_sz):
    # Permuting batch_size and n_steps
    x = tf.transpose(x, (1, 0, 2))
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, (T*batch_sz, D))
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.split(0, T, x) # v0.1
    x = tf.split(x, T) # v1.0
    # print "type(x):", type(x)
    return x

class SimpleRNN:
    def __init__(self, M):
        self.M = M # hidden layer size

    def fit(self, X, Y, batch_sz=20, learning_rate=10e-1, mu=0.99, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
        N, T, D = X.shape # X is of size N x T(n) x D
        K = len(set(Y.flatten()))
        M = self.M
        self.f = activation

        # initial weights
        # note: Wx, Wh, bh are all part of the RNN unit and will be created
        #       by BasicRNNCell
        Wo = init_weight(M, K).astype(np.float32)
        bo = np.zeros(K, dtype=np.float32)

        # make them tf variables
        self.Wo = tf.Variable(Wo)
        self.bo = tf.Variable(bo)

        # tf Graph input
        tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
        tfY = tf.placeholder(tf.int64, shape=(batch_sz, T), name='targets')

        # turn tfX into a sequence, e.g. T tensors all of size (batch_sz, D)
        sequenceX = x2sequence(tfX, T, D, batch_sz)

        # create the simple rnn unit
        rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)

        # Get rnn cell output
        # outputs, states = rnn_module.rnn(rnn_unit, sequenceX, dtype=tf.float32)
        outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

        # outputs are now of size (T, batch_sz, M)
        # so make it (batch_sz, T, M)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T*batch_sz, M))

        # Linear activation, using rnn inner loop last output
        logits = tf.matmul(outputs, self.Wo) + self.bo
        predict_op = tf.argmax(logits, 1)
        targets = tf.reshape(tfY, (T*batch_sz,))

        cost_op = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=targets
          )
        )
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)

        costs = []
        n_batches = N / batch_sz

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                n_correct = 0
                cost = 0
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

                    _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})
                    cost += c
                    for b in range(batch_sz):
                        idx = (b + 1)*T - 1
                        n_correct += (p[idx] == Ybatch[b][-1])
                
                if i % 10 == 0:
                    print ("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                if n_correct == N:
                    print ("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                    break
                costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename, activation):
        # TODO: would prefer to save activation to file too
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wxz = npz['arr_5']
        Whz = npz['arr_6']
        bz = npz['arr_7']
        Wo = npz['arr_8']
        bo = npz['arr_9']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)
        return rnn

    def generate(self, word2idx):
        # convert word2idx -> idx2word
        idx2word = {v:k for k,v in word2idx.items()}
        V = len(word2idx)

        # generate 4 lines at a time
        n_lines = 0

        # why? because using the START symbol will always yield the same first word!
        X = [ 0 ]
        while n_lines < 4:
            # print "X:", X
            PY_X, _ = self.predict_op(X)
            PY_X = PY_X[-1].flatten()
            P = [ np.random.choice(V, p=PY_X)]
            X = np.concatenate([X, P]) # append to the sequence
            # print "P.shape:", P.shape, "P:", P
            P = P[-1] # just grab the most recent prediction
            if P > 1:
                # it's a real word, not start/end token
                word = idx2word[P]
                print(word,)
            elif P == 1:
                # end token
                n_lines += 1
                X = [0]
                print('')

def train_poetry(artist, d, m, learning_rate, activation, epochs):
    # students: tanh didn't work but you should try it
    sentences, word2idx, _ = map_lyrics(artist=artist)
    rnn = SimpleRNN(d)
    rnn.fit(sentences, learning_rate=learning_rate, show_fig=True, activation=activation, epochs=epochs)
    rnn.save('Models/{artist:s}RRNN_D{d:s}_M{m:s}_e{epochs:s}_relu.npz'.format(
        artist=artist, d=d, m=m, epochs=epochs))

def generate_poetry(artist, d, m, epochs):
    sentences, word2idx, _ = map_lyrics(artist=artist)
    rnn = SimpleRNN.load(
        'Models/{artist:s}RRNN_D{d:s}_M{m:s}_e{epochs:s}_relu.npz'.format(
            artist=artist, d=d, m=m, epochs=epochs), 
        T.nnet.relu)
    rnn.generate(word2idx)

if __name__ == '__main__':
    artist = 'foo_fighters'
    d = 50
    m = 50
    learning_rate = 0.1
    epochs = 200
    activation=tf.nn.relu

    train_poetry(artist, d, m, learning_rate, activation, epochs)
    generate_poetry(artist, d, m, epochs)

