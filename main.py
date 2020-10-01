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
#infile = 'C:/Users/Wyatt Green/Downloads/Tom_Step1_out2.csv'
infile = 'C:/Users/Wyatt Green/Downloads/spamEmails.csv'
#data = "V1V2V3"
#labels = "V4"
data = 'v2'
labels = 'v1'



class Classifier():
    def __init__(self):
        #train_set and test_set are the entire sets of the dataset, including variables and class label
        self.train_set, self.test_set = self.load_data()
        #train_counts and test_counts are the frequency of words in the dataset
        self.train_counts, self.test_counts = self.vectorize()
        self.train_encodedLbls, self.test_encodedLbls = self.encodeClassLabels()
        self.classifier = self.train_model()

    def load_data(self):
        df = pd.read_csv(infile, header=0, error_bad_lines=False, sep = '|', engine = 'python')
        train_set, test_set = train_test_split(df, test_size=.3)
        return train_set, test_set
    
    def vectorize(self):
        vectorizer = TfidfVectorizer(min_df=5,
                                     max_df = 0.8,
                                     sublinear_tf=True,
                                     ngram_range = (1,2),
                                     use_idf=True)
        train_counts = vectorizer.fit_transform(self.train_set[data])
        test_counts = vectorizer.transform(self.test_set[data])
        return train_counts, test_counts 
    
    def encodeClassLabels(self):
        trainTargets = preprocessing.LabelEncoder().fit_transform(self.train_set[labels])
        testTargets = preprocessing.LabelEncoder().fit_transform(self.test_set[labels])
        return trainTargets, testTargets

    def train_model(self):
        classifier = MultinomialNB()
        
        #trainTargets = self.train_set[labels]
        #labelEncoding = preprocessing.LabelEncoder()
        #encodedLabels = labelEncoding.fit_transform(trainTargets)

        classifier.fit(self.train_counts, self.train_encodedLbls)
        return classifier

    def evaluate(self):
        
        #testTargets = self.test_set[labels]
        #labelEncoding = preprocessing.LabelEncoder()
        #encodedLbls = labelEncoding.fit_transform(testTargets)
        
        predictions = self.classifier.predict(self.test_counts)
        print (classification_report(self.test_encodedLbls, predictions))
        print ("The accuracy score is {:.2%}".format(accuracy_score(self.test_encodedLbls, predictions)))
        counter = 0
        for x in range(len(self.test_encodedLbls)):
            if(self.test_encodedLbls[x] == predictions[x]):
                counter = counter + 1
        print("Number of labels predicted correctly:" , counter)
        print("Number of total predictions:" , len(predictions))

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
