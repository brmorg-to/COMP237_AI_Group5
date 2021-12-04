#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 10:52:33 2021

@author: brunomorgado
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import re
import string


plt.style.use('seaborn-dark')

#Loading the CSV file
posts = pd.read_csv('/Users/brunomorgado/Dropbox/Education/Mac_Education/Centennial_College/Third_semester/AI/Assignments/Group_project/Youtube05-Shakira.csv')

#Initial exploration of the dataframe
pd.set_option('display.max_columns', None)

posts.head()

posts.tail()

posts.info()

# Using a heatmaps to better visualize the integrity of the dataframe.
# The information provided by posts.info() is visualy confirmed. There are no missing data in this dataframe.
plt.figure(figsize = (16,8))
sns.heatmap(posts.isnull(), cmap = 'viridis', cbar = False)

#Columns 'COMMENT_ID nd DATE are droped. Both seem not to carry any pattern that will influence in the prediction of the target variable.
posts.drop(['COMMENT_ID', 'DATE'], axis=1, inplace = True)

posts.head()

# Checking for imbalanced target variable.
posts.CLASS.value_counts()

target_freq = pd.Series(posts.CLASS).value_counts(normalize=True)

print(target_freq)

# Plot the relative frequency of the binary target variable
plt.figure(figsize = (10,5))
sns.barplot(x = target_freq.index, y = target_freq)
plt.ylabel('Relative Frequency')
plt.xlabel('Classes')

'''It seems that the response classe is quite well balanced. In a preliminary analysis, we may conclude that
it might not be of any value to "tweak" the weights of the labels (oversample any of the target variables).'''

#Split the dataset into features and target variable.
posts_features = posts[['AUTHOR', 'CONTENT']]
posts_target = posts.CLASS   

posts_features.head()
posts_features.info()

#In orther to be able to feed the features to the model, we will combine the to selected features in one single variable.
posts_combined = posts_features['AUTHOR'] + ' ' + posts_features['CONTENT']
posts_combined=pd.DataFrame(posts_combined, columns=["Posts"])
posts_combined.head()
posts_combined.shape
posts_combined.info()


#Function to remove any undesired extra space between the words
def clean_text_round(text):
   
# =============================================================================
#     text = text.lower()
#     text = re.sub('\[.*\]', ' ', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = re.sub('\w*\d\w*', ' ', text)
#     text = re.sub('['' “” ... •]', ' ', text)
#     text = re.sub('\n', '', text)
# =============================================================================
    text = re.sub(' +', ' ', text)
    # text = text.replace(u'\xa0', u'')
    return text


#Cleaning the posts
posts_combined['Posts'] = posts_combined['Posts'].apply(clean_text_round)

posts_combined.head()

#Export the combined posts for better visualization in excel
posts_combined.to_csv('/Users/brunomorgado/Dropbox/Education/Mac_Education/Centennial_College/Third_semester/AI/Assignments/Group_project/posts_combined.csv')


#Concatenating feature and target for split into train and test.
posts_processed = pd.concat([posts_combined, posts_target], axis = 1)

posts_processed.head()

posts_processed.shape

#Set the seed
np.random.seed(237)

#Shuffle the dataframe
posts_processed = posts_processed.sample(frac=1).reset_index(drop=True)

#Set the split proportion
split_index = int(0.75 * len(posts_processed))

#Splitting the dataframe into 75% train and 25% test
posts_train = posts_processed[:split_index]
posts_test = posts_processed[split_index:]

posts_train.shape
posts_test.shape

#Split the dataframes into feature and target
posts_train_X = posts_train.drop(['CLASS'], axis = 1)
posts_train_X.shape
posts_train_y = posts_train['CLASS']
posts_train_y.shape

posts_test_X = posts_test.drop(['CLASS'], axis = 1)
posts_test_y = posts_test['CLASS']
posts_test_y.shape

#Using CountVectorizer to transform the posts into a vector on the basis of the frequency of each word that occurs in the text.
cv = CountVectorizer(stop_words='english')
features_cv_train = cv.fit_transform(posts_train_X.Posts)

#Using Tfidf (term frequency-inverse document frequency) to get a 'score' that measures how relevant a word is to a document in a collection of documents.
tfidf = TfidfTransformer()
features_tfidf = tfidf.fit_transform(features_cv_train)
type(features_tfidf)
# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(features_tfidf, posts_train_y)

#Use cross-validation (5-fold) to evaluate the model's accuracy.  
scores = cross_val_score(classifier, features_tfidf, posts_train_y, cv=5)
print(scores.mean())

type(posts_test_X)
#Resetting the index
posts_test_X=posts_test_X.reset_index(drop=True)

posts_test_X



# Transform test data using count vectorizer
features_cv_test = cv.transform(posts_test_X.Posts)
type(features_cv_test)

# Transform vectorized test data using tfidf transformer
input_tfidf = tfidf.transform(features_cv_test)
type(input_tfidf)


# Predict the output categories
predictions = classifier.predict(input_tfidf)

# Print the outputs
print(predictions)

predictions.shape
#Print classification report
print(classification_report(posts_test_y, predictions))

#Confusion Matrix
cf_matrix_shakira = confusion_matrix(posts_test_y, predictions)
print(cf_matrix_shakira)

#Plot the confusion Matrix
class_names=[0,1]
fig, ax = plt.subplots(figsize=(10,6))
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(cf_matrix_shakira, annot=True,cmap="PuBu" ,fmt='g')
ax.set_ylim([0,2])
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix - Naive Bayes',fontsize=20)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


'''precision means what percentage of the positive predictions made were actually correct.

TP/(TP+FP)

'''

'''Recall in simple terms means, what percentage of actual positive predictions were correctly classified by the classifier.

TP/(TP+FN)

'''

'''F1 score can also be described as the harmonic mean or weighted average of precision and recall.

2x((precision x recall) / (precision + recall))

'''