
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adagrad

from src.util.utils import build_skip_gram

class Word2Vec:
  '''
    Class which contains the word -> vector model architecture implemented
    with keras
  '''
  def __init__(self):
    self.model = None 

  def build(self, emb_dim, vocab_size, learn_rate):
    '''
      Builds the model given hyper-parameters
    '''
    std_dev = 1.0 / vocab_size # standard deviation for rand init
    initializer = RandomNormal(mean=0.0, stddev = std_dev, seed = 42) # weight init

    # input layer
    word_input = Input(shape = (1,), name = 'words_input')
    # embedding layer
    embed_layer = Embedding(
      input_dim = vocab_size, # vocab-size of the sequence passed
      output_dim = emb_dim, # the embedding dimension setted by the hyper-params
      input_length = 1, # we are just dealing with one huge sequence
      name = 'word_embeddings',
      embeddings_initializer = initializer, # init weights defined by RandomNormal
    )(word_input)

    # context layers
    # context input layer
    context_input = Input(shape=(1,) , name = 'context_input')
    # embeddings context layer
    context_embeds = Embedding(
      input_dim = vocab_size,
      output_dim = emb_dim,
      input_length = 1,
      name = 'context_embeddings',
      embeddings_initializer = initializer 
    )(context_input)

    # merge the input with the context
    merged = dot([embed_layer, context_embeds], axes = 1, normalize = False)
    merged = Flatten()(merged)
    output = Dense(1, activation='sigmoid', name="output")(merged)

    # define the optimizer
    optimizer = Adagrad(learn_rate)
    # define the model after the merge
    model = Model(inputs=[word_input, context_input], outputs=output)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    self.model = model # save into the obj

  def train(self, sequence, window_size, negative_samples, batch_size, epochs):
    # balancing the negative samples
    negative_weight = 1.0 / negative_samples
    class_weight = {1: 1.0, 0: negative_weight}

    sequence_length = len(sequence) # get the length of all the sequence
    approx_steps_per_epoch = (sequence_length * (
      window_size * 2.0) + sequence_length * negative_samples) / batch_size
    seed = 42
    batch_iterator = build_skip_gram(sequence, window_size, negative_samples, batch_size, seed)
    #print(next(batch_iterator))
    self.model.fit_generator(
      batch_iterator,
      steps_per_epoch=approx_steps_per_epoch,
      epochs=epochs,
      verbose=True,
      class_weight=class_weight,
      max_queue_size=100
    )
    print('---> Training Finished')

  def write_embeddings(self, path, idx2wrd, embeddings):
    '''
      Write embeddings method
    '''
    np_index = np.empty(shape=len(idx2wrd), dtype=object)
    for index, word in idx2wrd.items():
      np_index[index] = word
    
    df = pd.DataFrame(data=embeddings, index=np_index)
    df.to_csv(path, float_format="%.4f", header=False)
