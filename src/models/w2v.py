import numpy as np
import multiprocessing
# from tensorflow.keras.preprocessing.text import one_hot
# from tensorflow.keras import layers
# from tensorflow import keras
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

from src.preprocessing.spacy_trans import clean_corpus
from src.util.utils import load_data, build_vocab, build_word_indices, txt_to_wrd_seq
from src.keras.keras_models import Word2Vec

def tokenize_corpus():
	'''
		Loads the data and calls spacy tokenization
	'''
	train, test = load_data()
	X, y  = train['text'], train['target']
	
	# clean + tokenize
	cleaned_corpus = clean_corpus(X)

	return cleaned_corpus

def keras_w2v():
	'''
		w2v using keras as backend
	'''
	cleaned_corpus = tokenize_corpus() # clean + tokenize TODO check cleaning for \n\n characters
	vocab_dict = build_vocab(cleaned_corpus) # build the vocab
	wrd2idx, idx2wrd = build_word_indices(vocab_dict) # build wrd2idx + idx2word dicts
	wrd_seq = txt_to_wrd_seq(cleaned_corpus, wrd2idx) # build the int seqs of words
	wrd_seq = np.asarray(wrd_seq, dtype=int) # transform into a numpy array

	vocab_size = len(vocab_dict)
	 
	Word_to_Vec_model = Word2Vec() # call the obj
	Word_to_Vec_model.build(100, vocab_size, 0.1) # build the model
	Word_to_Vec_model.train(wrd_seq, 3, 10, 100, 1)

	# write embeddings
	emb_layer = Word_to_Vec_model.model.get_layer("word_embeddings").get_weights()[0]
	Word_to_Vec_model.write_embeddings('./data/embeddings.csv', idx2wrd, emb_layer)
	
def gensim_w2v():
	'''
		w2v using the gensim lib
	'''
	cleaned_corpus = tokenize_corpus()
	
	# a Phraser takes a list of lists of words as input
	phrases = Phrases(cleaned_corpus, min_count=30, progress_per=10)
	bigram = Phraser(phrases) # construct the bigram object form the extracted phrases
	
	sentences = bigram[cleaned_corpus] # this will construct words like northern california into northern_california

	n_cores = multiprocessing.cpu_count() # count the number of cores in our computer
	w2vec = Word2Vec(
		min_count=20,
		window=2,
		size=300,
		sample=6e-5,
		alpha=0.03,
		min_alpha=0.0007, 
    negative=20,
    workers=n_cores-1
	)
	w2vec.build_vocab(sentences) # build the vocab given the sentences
	w2vec.train(sentences, total_examples=w2vec.corpus_count, epochs=30, report_delay=1) # train

	emb_matrix = w2vec[w2vec.wv.vocab] # save for viz maybe?
	mean_vector = np.mean(emb_matrix, axis=0) # this will be used for the <UNK> tokens on test data

if __name__ == '__main__':
  #gensim_w2v()
	keras_w2v()
