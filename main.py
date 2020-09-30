import csv
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#infile = 'C:/Users/wgree/Documents/Work/MultiNB/Tom_Step1_out2.csv'
infile = 'C:/Users/Wyatt Green/Downloads/Tom_Step1_out2.csv'
data = "V1V2V3"
labels = "V4"


class Classifier():
    def __init__(self):
        self.train_set, self.test_set = self.load_data()
        self.counts, self.test_counts = self.vectorize()
        self.classifier = self.train_model()

    def load_data(self):
        df = pd.read_csv(infile, header=0, error_bad_lines=False, sep = '|')
        train_set, test_set = train_test_split(df, test_size=.3)
        return train_set, test_set
    
    def vectorize(self):
        vectorizer = TfidfVectorizer(min_df=5,
                                     max_df = 0.8,
                                     sublinear_tf=True,
                                     ngram_range = (1,2),
                                     use_idf=True)
        counts = vectorizer.fit_transform(self.train_set[data])
        test_counts = vectorizer.transform(self.test_set[data])
        return counts, test_counts

    def train_model(self):
        classifier = MultinomialNB()
        trainTargets = self.train_set[labels]
        labelEncoding = preprocessing.LabelEncoder()
        encodedLabels = labelEncoding.fit_transform(trainTargets)
        classifier.fit(self.counts, encodedLabels)
        return classifier

    def evaluate(self):
        testTargets = self.test_set[labels]
        labelEncoding = preprocessing.LabelEncoder()
        encodedLbls = labelEncoding.fit_transform(testTargets)
        predictions = self.classifier.predict(self.test_counts)
        print (classification_report(encodedLbls, predictions))
        print ("The accuracy score is {:.2%}".format(accuracy_score(encodedLbls, predictions)))

    def classify(self, input):
        input_text = input

        input_vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 ngram_range = (1,2),
                                 use_idf=True)
        input_counts = input_vectorizer.transform(input_text)
        predictions = self.classifier.predict(input_counts)
        print(predictions)

myModel = Classifier()

#text = ['I like this I feel good about it', 'give me 5 dollars']
#myModel.classify(text)
myModel.evaluate()
