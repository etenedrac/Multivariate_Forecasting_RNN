"""
.. module:: TextGenerator

TextGenerator
*************

:Description: TextGenerator

    

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:26 

"""

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
import numpy as np
import random
import gzip
import argparse
import time

def sample(preds, temperature=1.0):
    """
    helper function to sample an index from a probability array

    :param preds:
    :param temperature:
    :return:
    """

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(seed, numlines, gfile, wseed=False):
    """
    Generates a number of lines (or at most 1000 characters) using the given seed
    :param seed:
    :param lines:
    :return:
    """
    generated = ''
    gprinted = ''
    sentence = seed
    generated += sentence

    nlines = 0
    for i in range(1000):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        gprinted += next_char

        sentence = sentence[1:] + next_char
        # Count the number of lines generated
        if next_char == '\n':
            nlines += 1
        if nlines > numlines:
            break

    if wseed:
        gfile.write(seed + gprinted)
    else:
        gfile.write(gprinted)
    gfile.write('\n')
    gfile.flush()


def random_seed(chars, nchars):
    """
    Generates a random string
    :param nchars:
    :return:
    """
    s = ""
    for i in range(nchars):
        s += chars[random.randint(0, len(chars) - 1)]

    return s


myseeds = ["behold the merry bride,\nwhite dress with yellow flowers,\nbright smile with sunny red,\nsweet my love ",
           "land and trees cast sunny shadows,\nchildren laughter, merry sound,\nyellow birds wear long feathers,\n"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--progress', help="Saves the weights of the model each iteration", action='store_true', default=False)
    parser.add_argument('--save', help="Saves the last model", action='store_true', default=False)
    parser.add_argument('--datafile', help="Number of the datafile to use [1,2,3,4]", type=int, default=1)
    parser.add_argument('--iterations', help="Number of iterations to train", type=int, default=1)
    parser.add_argument('--neurons', help="Number of neurons in the recurrent layers", type=int, default=256)
    parser.add_argument('--layers', help="Number of layers", type=int, default=1)
    parser.add_argument('--dropout', help="Recurrent dropout", type=float, default=0.0)
    parser.add_argument('--exlen', help="Length of the examples", type=int, default=50)
    parser.add_argument('--step', help="Step of the moving window for generating examples", type=int, default=3)

    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2

    print("Starting:", time.ctime())

    ############################################
    # Data

    if args.datafile not in [1,2,3,4]:
        raise NamedError('No such data file')
    else:
        path = f'poetry{args.datafile}.txt.gz'

    text = gzip.open(path, 'rt').read().lower().replace('\ufeff', ' ')
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = args.exlen
    step = args.step
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    # Vectorizes the sequences with one hot encoding
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    ############################################
    # Model
    print('Build model...')

    RNN = LSTM  # GRU
    lsize = args.neurons
    nlayers = args.layers
    dropout = args.dropout

    model = Sequential()
    if nlayers == 1:
        model.add(RNN(lsize, input_shape=(maxlen, len(chars)), implementation=impl, recurrent_dropout=dropout))
    else:
        model.add(
            RNN(lsize, input_shape=(maxlen, len(chars)), implementation=impl, recurrent_dropout=dropout, return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(lsize, implementation=impl, recurrent_dropout=dropout, return_sequences=True))
        model.add(RNN(lsize, implementation=impl, recurrent_dropout=dropout))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    ############################################
    # Training
    # optimizer = RMSprop(lr=0.01)
    #optimizer = SGD(lr=0.05, momentum=0.95)
    model.compile(loss='categorical_crossentropy', optimizer="adam")

    bsize = 2048
    iterations = args.iterations
    epoch_it = 10

    # File for saving the generated text each iteration
    gfile = open('tgenerated-F%s-ML%d-S%d-NL%d-D%3.2f-BS%d.txt' % (path.split('.')[0], maxlen, lsize, nlayers, dropout, bsize), 'w')

    # train the model, output generated text after each iteration
    for iteration in range(iterations):
        print()
        print('-' * 50)
        print('Iteration %d - %s' %((iteration + 1), time.ctime()))
        model.fit(X, y,
                  batch_size=bsize,
                  epochs=epoch_it,
                  verbose=verbose)

        gfile.write('-' * 50)
        gfile.write('\n')
        gfile.write(time.ctime())
        gfile.write('\n')
        gfile.write('Iteration %d n' % (iteration + 1))
        seed = random_seed(chars, maxlen)
        for diversity in [0.2, 0.4, 0.8, 1.0]:
            gfile.write('\n\n')
            gfile.write('DIV = %3.2f\n\n' % diversity)
            generate_text(seed, numlines=10, gfile=gfile, wseed=False)
        if args.progress:
            model.save(f"./textgen-T{args.datafile}-N{lsize}-L{nlayers}-D{dropout}-{time.strftime('%Y%m%d%H%M%S')}.h5")
    gfile.close()
    if args.save:
        model.save(f"./textgen-T{args.datafile}-N{lsize}-L{nlayers}-D{dropout}-{time.strftime('%Y%m%d%H%M%S')}.h5")
    print()
    print("Ending:", time.ctime())
