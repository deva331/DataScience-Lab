import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import ngrams
sentence=input("Enter the Sentence")
n=int(input("Enter the value of n"))
n_grams=ngrams(sentence.split(),n)
print("ngrams.printing")
for grams in n_grams:
 print(grams)
from nltk import word_tokenize,sent_tokenize
print("tokens printing")
print(word_tokenize(sentence))
print(sent_tokenize(sentence))
from nltk import pos_tag
tokenized_text=word_tokenize(sentence)
tags=tokens_tag=pos_tag(tokenized_text)
print(tags)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
ps=PorterStemmer()
for w in tokenized_text:
 print(w,":",ps.stem(w))
                    