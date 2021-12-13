import numpy as np
from keras.layers import Input,Embedding,Lambda
from keras.models import Model,load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import keras.backend as K
import pandas as pd
import json

word_size = 300  # số  chiều
window = 5  
nb_negative = 15  # Number of samples for random negative sampling
min_count = 0  # Words with frequency less than min_count will be discarded
nb_worker = 1  # The number of concurrent reading data
nb_epoch = 20  #The number of iterations, due to the use of adam, the effect of the number of iterations 1 to 2 is quite good
subsample_t = 1e-5  # Words with word frequency greater than subsample_t will be downsampled,
                    # which is an effective solution to improve speed and word vector quality
nb_sentence_per_batch = 30  # At present, the unit of sentence is used as a batch, and how many sentences are used as a batch 
                            #(in this way, it is easy to estimate the steps parameter in the training process.
                            # Also note that the number of samples is proportional to the number of words.)



def getdata():
    data = pd.read_csv('../datasets/HDFS/templates.csv').values
    templates = data[:,1]
    label = data[:,0]   #label ở đây là event id đó
    sentences = []
    for s in templates:
        sentences.append(s.split())
    return label,templates, sentences



def bulid_dic(sentences):  
    words = {}  # Word frequency table
    nb_sentence = 0  # Total sentences
    total = 0.  # Total word frequency

    for d in sentences:
        nb_sentence += 1
        for w in d:
            if w not in words:
                words[w] = 0
            words[w] += 1
            total += 1
        if nb_sentence % 100 == 0:
            pass

    words = {i: j for i, j in words.items() if j >= min_count}  # Truncated word frequency
    id2word = {i + 1: j for i, j in enumerate(words)}  # ID to word mapping, 0 means UNK
    word2id = {j: i for i, j in id2word.items()}  # Word to id mapping
    nb_word = len(words) + 1  # Total number of words (counting the fill symbol 0)

    subsamples = {i: j / total for i, j in words.items() if j / total > subsample_t}
    subsamples = {i: subsample_t / j + (subsample_t / j) ** 0.5 for i, j in
                  subsamples.items()}  # This downsampling formula is based on the source code of word2vec
    subsamples = {word2id[i]: j for i, j in subsamples.items() if j < 1.}  # Downsampling table
    return nb_sentence, id2word, word2id, nb_word, subsamples


def data_generator(word2id, subsamples, data):  # Training data generator
    x, y = [], []
    _ = 0
    for d in data:
        d = [0] * window + [word2id[w] for w in d if w in word2id] + [0] * window
        r = np.random.random(len(d))
        for i in range(window, len(d) - window):
            if d[i] in subsamples and r[i] > subsamples[d[i]]:  # Direct skip that satisfies the downsampling condition
                continue
            x.append(d[i - window:i] + d[i + 1:i + 1 + window])
            y.append([d[i]])
        _ += 1
        if _ == nb_sentence_per_batch:
            x, y = np.array(x), np.array(y)
            z = np.zeros((len(x), 1))
            return [x, y], z


def build_w2vm(word_size, window, nb_word, nb_negative):
    K.clear_session()  # Clear the previous model to avoid filling up the memory
    # CBOW 
    input_words = Input(shape=(window * 2,), dtype='int32')
    input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)  # CBOW model, which directly sums the context word vector

    # Construct a random negative sample to form a sample with the target
    target_word = Input(shape=(1,), dtype='int32')
    negatives = Lambda(lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, nb_word, 'int32'))(target_word)
    samples = Lambda(lambda x: K.concatenate(x))([target_word, negatives])  # Sampling is constructed, and negative samples are randomly selected. 
                                                                            #A negative sample may also be a positive sample, but the probability is small.

    # Do Dense and softmax only in the sample
    softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
    softmax_biases = Embedding(nb_word, 1, name='b')(samples)
    softmax = Lambda(lambda x:
                     K.softmax((K.batch_dot(x[0], K.expand_dims(x[1], 2)) + x[2])[:, :, 0])
                     )([softmax_weights, input_vecs_sum, softmax_biases])  # Use the Embedding layer to store parameters, 
                     #and use the K back-end to implement matrix multiplication to reproduce the function of the Dense layer

    # Note that when we constructed the sampling, we put the target in the first place, 
    #that is, the target id of softmax is always 0, which can be seen from the writing of the z variable in data_generator
    model = Model(inputs=[input_words, target_word], outputs=softmax)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Please note that sparse_categorical_crossentropy is used instead of categorical_crossentropy
    model.summary()
    return model


if __name__ == '__main__':
    #word2vector
    label,templates, sentences = getdata()
    nb_sentence, id2word, word2id, nb_word, subsamples = bulid_dic(sentences)
    ipt, opt = data_generator(word2id, subsamples, templates) # Construct training data
    model = build_w2vm(word_size, window, nb_word, nb_negative) # model
    model.fit(ipt, opt,steps_per_epoch=int(nb_sentence / nb_sentence_per_batch),epochs=nb_epoch)
    model.save('hdfsword2vec.h5')
    embeddings = model.get_weights()[0]
    normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1,1))**0.5 #The word vector is normalized, i.e. the modulus is 1embeddings[0]embeddings[0]

    #Save sentence vector
    vector_json={}
    for i in range(0,len(sentences)):
        vector = []
        for ii in sentences[i]:
            vector.append(normalized_embeddings[word2id[ii]])
        vector_json.update({(i+1):list(np.float64(np.sum(vector,axis=0)))})
    json_str = json.dumps(vector_json)
    with open('../datasets/HDFS/hdfs_vector.json', 'w') as json_file:
        json_file.write(json_str)
