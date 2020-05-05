import string
import re
import pandas as pd 
import os
import random
import numpy as np

from collections import defaultdict

def url_strip(document):
  '''
    removes the urls from the tokens
    Args:
      document : string 
        word(s) document

    Returns:
      document : string
        document without urls
  '''
  url = re.compile(r'https?://\S+|www\.\S+') # reegular expression for an url
  return url.sub('', document)

def remove_html(document):
  '''
    Removes html strings
  '''
  html = re.compile(r'<.*?>')
  
  return html.sub('', document)

def remove_emoji(document):
  emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
  
  return emoji_pattern.sub(r'', document)

def remove_punct(document):
  '''
    Removes puntuation #, @ etc
  '''
  return document.translate(str.maketrans('', '', string.punctuation))

def clean_document(document):
  document = remove_html(url_strip(document)).lower() # remove urls
  document = remove_html(document) # remove htmls
  document = remove_emoji(document) # remove emojis <- this can be handled in spacy directly
  document = remove_punct(document) # remove punctuation
  document = document.strip().rstrip() # remove front and end whitespace
  document = re.sub(' +', ' ',document) # remove multiple whitespace
  return document
  
def load_data():
  '''
    Loads the data located in the (root of theproject)/data folder

    Returns:
      train : pd dataframe 
      test : pd dataframe
  '''
  # walk the data folder
  file_paths = []
  for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        name = str(os.path.join(dirname, filename))
        file_paths.append(name)

  train = pd.read_csv(file_paths[1])
  test = pd.read_csv(file_paths[2])

  return train, test

def build_vocab(cleaned_corpus):
  '''
    Builds the vocabulary given a cleaned corpus
  '''
  word_freq = defaultdict(int)
  for sentence in cleaned_corpus:
    for word in sentence:
      word_freq[word] += 1 # sum 1 for each word occurance

  return word_freq

def build_word_indices(word_dict):
  wrd2idx = dict() # word to index dictionary
  idx2wrd = dict()  # index to word dictionary

  for idx, wrd in enumerate(word_dict):
    wrd2idx[wrd] = idx
    idx2wrd[idx] = wrd
  
  return wrd2idx, idx2wrd

def txt_to_wrd_seq(corpus, wrd2idx):
  '''
    transforms a corpus into a sequence of previously encoded integers
  '''
  sequence = [] # define empty sequence
  for document in corpus:
    for word in document:
      sequence.append(wrd2idx[word]) # append the int representing that word

  return sequence

def build_iterator(seq, windows_size, neg_samples, seed):
  '''
    Build the skipgram generator
    returns (word, context, label) at each iteration
  '''
  random.seed = seed # set the seed
  seq_len = seq.shape[0] 
  epoch = 0
  i = 0 
  while True:
    # define start and end of the window
    window_start = max(0, i - windows_size)
    window_end = min(0, i + windows_size + 1)

    # loop over the length of the windows to get the positive samples
    for j in range(window_start, window_end): 
      if i != j:
        word = seq[i]
        context = seq[j]
        yield (word, context, 1)
    
    # loop over the neg samples (random word sample)
    for negative in range(neg_samples):
      # get a random negative sample
      random_int = random.randrange(1, seq_len)
      word = seq[i]
      context = seq[random_int]
      yield (word, context, 0)

    i += 1
    if i == seq_len:
      epoch += 1
      print("iterated %d times over data set", epoch)
      i = 0

def build_skip_gram(sequence, windows_size, negative_samples, batch_size, seed):
  ''''
    Builds the batch iterator
  '''
  iterator = build_iterator(sequence, windows_size, negative_samples, seed)
  
  words = np.empty(shape=batch_size, dtype=int)
  contexts = np.empty(shape=batch_size, dtype=int)
  labels = np.empty(shape=batch_size, dtype=int)
  
  while True:
      for i in range(batch_size):
        word, context, label = next(iterator)
        words[i] = word
        contexts[i] = context
        labels[i] = label
      yield ([words, contexts], labels)

if __name__ == '__main__':
  main()