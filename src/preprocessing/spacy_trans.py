import pandas as pd
import sys
import spacy

from src.util.utils import load_data, clean_document

def clean_corpus(corpus):
  nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # for now disable parser and named entity recog
  en_stop_words = list(spacy.lang.en.stop_words.STOP_WORDS) # define english stop words
  
  cleaned_corpus = [spacy_tokenizer(doc, nlp) for doc in corpus]

  return cleaned_corpus

def lematize(document):
  '''
    Lematizes the document (tweet)
    Args:
      document : doc
        document object from the spacy library
    Returns:
      tokens : list<doc>
        list of lematized tokens
  '''
  tokens = []
  for token in document:
    # append only if the token is not a stop word
    if not token.is_stop:
      if token.lemma_ != '-PRON-':
        tokens.append(token.lemma_) # if the lemma is not a pronoun then append
      else:
        tokens.append(token.lower_) # just append the string rep as is
  
  return tokens
   
def spacy_tokenizer(document, nlp):
  '''
    tokenizes a document using spacy, cleans text, lematizes
    and removes stop words

    Args:
      document : string
        document string
      nlp : spacy nlp
        nlp object that contains all the funcs for lenguare processing
    Returns:
      tokenized_document : list<spacy<token>>
        list of tokenized and cleaned tokens
  '''
  doc = clean_document(document)
  doc = nlp(doc)
  tokenized_document = lematize(doc)

  return tokenized_document

if __name__ == '__main__':
  for path in sys.path:
    print(path)

  main()
