#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:31:22 2021

@author: group-2
"""

import pandas as pd
import nltk

#loading data
filePath = "/Users/sambhu/Desktop/"
fileName= "Youtube02-KatyPerry.csv"
fullPath = filePath+fileName
KatyPerry_comments = pd.read_csv(fullPath)

#data exploration
#shape of data
print("Shape of data: " )
print("---------------------------------------")
print(KatyPerry_comments.shape)
print("First few records: " )
print("----------------------------------------")
print(KatyPerry_comments.head(3))
print("Datatype of columns: " )
print("-----------------------------------------")
print(KatyPerry_comments.dtypes)
print("check null data: " )
print("-----------------------------------------")
print(KatyPerry_comments.isna().any().sum())

#data pre-processing:
#dropping unnecessary columns
kattyPerry_comments=KatyPerry_comments.drop(columns=['COMMENT_ID','AUTHOR','DATE'])
print(kattyPerry_comments.tail(5))

# 6. Use pandas.sample to shuffle the dataset, set frac =1
kattyPerry_comments=KatyPerry_comments.sample(frac = 1)

#using count vectorization method: 
from sklearn.feature_extraction.text import CountVectorizer
#count vectorizer is a method to convert text to numerical data.
#setting lowercase to true to make sure all the words are converted to lower case
#setting stop_words to english to remove stopwords(These words are unnecessary during classification)  
coun_vect = CountVectorizer(lowercase=True,stop_words='english')
count_matrix = coun_vect.fit_transform(kattyPerry_comments['CONTENT'])

print(coun_vect.vocabulary_) 
 
#4new shape of the data
 
print("Shape of Output")
print(count_matrix.shape) 

#5Term Frequency times inverse Document frequency
# Create the tf-idf transformer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(count_matrix)


from sklearn.naive_bayes import MultinomialNB

# 7. split your dataset into 75% for training and 25% for testing
num_train = int(0.75 * len(kattyPerry_comments))
x_train , x_test = train_tfidf[:num_train], train_tfidf[num_train:]
y_train , y_test = kattyPerry_comments['CLASS'][:num_train],kattyPerry_comments['CLASS'][num_train:]
#y_train.reshape(262,1)
# 8. Fit the training data into a Naive Bayes classifier
classifier = MultinomialNB().fit(x_train,y_train)


# 9. Cross validate the model on the training data using 5-fold 
# and print the mean results of model accuracy.
print("---------------------------------------")
print("Point No. 9" )

from sklearn.model_selection import cross_val_score
num_folds = 5
accuracy_values = cross_val_score(classifier, 
        x_train, y_train , scoring='accuracy', cv=num_folds)
print("Mean Model Accuracy: " + str(accuracy_values.mean()))
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

# 10. Test the model on the test data, print the confusion matrix and the accuracy of the model.
print("---------------------------------------")
print("Point No. 10" )

from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(x_test)
print(type(y_pred))
print(type(y_test))
y_act = y_test.values
print(type(y_act))

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print(tn, fp, fn, tp)

print("Accuracy: " + str(round((tn+tp)/(tn+fp+fn+tp), 2)))


# 11. Come up with 6 new comments (4 comments should be non spam and 2 comment spam)
# and pass them to the classifier and check the results.
print("---------------------------------------")
print("Point No. 11" )

input_data = [
    'This song is so cool. I just love the beat. Katy Perry is so amazing!', 
    'Call +111143123 for instant interest free loans.',
    'Why am I so late to find this song? its so GOOOOOOOD!',
    'This song is like fine wine! It get better and better every day',
    'Want to earn FREE MONEY, follow the link to earn $$$$$$$$',
    'I got a speed ticket this morning coming to work because I was so lost in the song'
]

input_tc = coun_vect.transform(input_data)
type(input_tc)
#print(input_tc)
# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
#print(input_tfidf)
# Predict the output categories
predictions = classifier.predict(input_tfidf) 

print(predictions)

for sent, category in zip(input_data, predictions):
    if category == 0:
        label = 'Non spam'
    else:
        label = 'Spam'
    print('\nInput:', sent, '\nPredicted category:', \
            category,'-', label)































