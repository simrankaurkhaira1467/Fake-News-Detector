"""


The words have to be encoded as integer of floating point values for use as input to a machine learning algorithm therfore we do
feature extraction.
sklearn lib offers easy to use tools to perform tokenization and feature extraction of a text data.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
#feature_extraction module is used to extract features in a format supported by ml algorithm from dataset.
#count_vectorizer does both tokenization and occurrence counting in a single class.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics 
import numpy as np

df = pd.read_csv("fake_or_real_news.csv")
print(df.shape)
print(df.head())
df = df.set_index("Unnamed: 0")
print(df.head())
y = df.label
# where numbering of news article is done that column is dropped in dataset
df.drop("label", axis=1)
#testing and training datasets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.70, random_state=53)
count_vectorizer = CountVectorizer(stop_words='english') #stop_words remove english words from data before making vectors.
# Learn the vocabulary dictionary of training data and return term-document matrix
count_train = count_vectorizer.fit_transform(X_train)
#transform training data into document term matrix.
count_test = count_vectorizer.transform(X_test)                       
# This removes words which appear in more than 70% of the articles
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
#fit tells transformation is based on which features.
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#transform replaces the missing values with a number.
tfidf_test = tfidf_vectorizer.transform(X_test)
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
np.memmap("fake_or_real_news.csv")
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
clf = MultinomialNB()   #used for descrete text count.... also deals with word count and calculates with it 
# Fit Naive Bayes classifier according to x, y
clf.fit(tfidf_train, y_train)
# Perform classification on an array of test vectors X.
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
recall=metrics.recall_score(y_test, pred)
score=score*100
print("ACCURACY:   %0.3f %%" % score)
print("recall:   %0.3f %%" % recall)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)
