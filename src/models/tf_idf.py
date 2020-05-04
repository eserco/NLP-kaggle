from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import spacy

from src.preprocessing.spacy_trans import lematize, clean_document, spacy_tokenizer
from src.util.utils import load_data

def main():
  train, test = load_data()
  X, y  = train['text'], train['target']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

  nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # for now disable parser and named entity recog
  en_stop_words = list(spacy.lang.en.stop_words.STOP_WORDS) # define english stop words
  
  tf_idf = TfidfVectorizer(tokenizer=lambda text: spacy_tokenizer(text, nlp))
  clf = LinearSVC() 

  pipe = Pipeline([
    ('tfidf', tf_idf),
    ('classify', clf) 
  ])
  
  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_test)
  print(classification_report(y_test, y_pred))

  # ---- test funcs
  # corpus_generator = (nlp(document) for document in corpus) # define generator
  # g5 = [next(corpus_generator) for _ in range(5)] # just 2 first tweets from the generator
  # tokenized_corpus = [lematize(doc) for doc in g5]
  
if __name__ == '__main__':
  main()