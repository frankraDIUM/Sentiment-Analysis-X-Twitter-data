#!/usr/bin/env python
# coding: utf-8


#Add libraries

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


#Load data

df = pd.read_csv('flood_tweets.csv')

#Preview data


df.head()


#Get more details of data


df.info()


#Count null entries

df.isnull().sum()


#Get columns

df.columns


#Create text only dataframe

text_df = df.drop(['id', 'author_username', 'author_location', 'author_description', 'author_created', 'author_followers', 'author_following', 'author_favourites', 'author_verified', 'date', 'retweets', 'likes', 'isRetweet'], axis=1)
text_df.head()


#View raw text data from X (Twitter)

print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")

#Get details on text dataframe

text_df.info()


#Define function and pass text data to process data into usable format

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


#Apply to text data

text_df.text = text_df['text'].apply(data_processing)


#Remove duplicates 

text_df = text_df.drop_duplicates('text')


#Perform stemming on words

stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


#Apply stem to data

text_df['text'] = text_df['text'].apply(lambda x: stemming(x))


#Preview processed data

text_df.head()


#Analyze data

print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


#Get updated details on data

text_df.info()


#Calculate text polarity with TextBlob

def polarity(text):
    return TextBlob(text).sentiment.polarity

#Add to data frame

text_df['polarity'] = text_df['text'].apply(polarity)


#Preview data frame with polarity

text_df.head(10)


#Add function for Sentiment

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


#Apply on data frame


text_df['sentiment'] = text_df['polarity'].apply(sentiment)


#Preview data frame

text_df.head()


#Visualize the distribution of data with plot

fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = text_df)


#Visualize data with pie chart

fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')


#View top five tweets for positive sentiment

pos_tweets = text_df[text_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
pos_tweets.head()


#Visualize positive tweets

text = ' '.join([word for word in pos_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show()



#View top five tweets in negative sentiment

neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending= False)
neg_tweets.head()


#Visualize negative tweets

text = ' '.join([word for word in neg_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative tweets', fontsize=19)
plt.show()


#View top five tweets with neutral sentiment

neutral_tweets = text_df[text_df.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending= False)
neutral_tweets.head()


#Visualize tweets

text = ' '.join([word for word in neutral_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()


# #Vectorize data

vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])


#Get features

feature_names = vect.get_feature_names_out()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features:\n {}".format(feature_names[:20]))


#Build model.Separate data into X and Y

X = text_df['text']
Y = text_df['sentiment']
X = vect.transform(X)


#Place data into testing and training data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#Print size of testing and training data

print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))


import warnings
warnings.filterwarnings('ignore')


#Load logistic regression model

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


#Print model accuracy
#Add Confusion matrix and Classification report

print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


#Visualize Confusion matrix

style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=logreg.classes_)
disp.plot()


#Import GridSearchCV

from sklearn.model_selection import GridSearchCV


#Add hyperparameter tuning to test model performance

param_grid={'C':[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)


GridSearchCV(estimator=LogisticRegression(),
             param_grid={'C': [0.001, 0.01, 0.1, 1, 10]})


#Print best parameter

print("Best parameters:", grid.best_params_)

y_pred = grid.predict(x_test)


#Get and print accuracy score

logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


#Print Confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))


#Run model with Support Vector Macine
#import Support Vector classifier

from sklearn.svm import LinearSVC


#Load classifier and fit data

SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)

#Add values for given test data. Calculate test accuracy

svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("test accuracy: {:.2f}%".format(svc_acc*100))


#Print confusion matrix and classification report

print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))


#Add hyperparameter tuning to test model performance

grid = {
    'C':[0.01, 0.1, 1, 10],
    'kernel':["linear","poly","rbf","sigmoid"],
    'degree':[1,3,5,7],
    'gamma':[0.01,1]
}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)


#Print best parameter

print("Best parameter:", grid.best_params_)


y_pred = grid.predict(x_test)


#Get and print accuracy score

logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


#Print confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))






