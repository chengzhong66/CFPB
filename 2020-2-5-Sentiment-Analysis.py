#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import swifter
import gc
import nltk 
import sklearn 
import collections
import sys
import itertools
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm


# In[2]:


df_CFPB = pd.read_csv("complaints.csv")


# In[3]:


df_CFPB.head()


# In[4]:


# Number of missing complaint narratives
no_miss_narr = df_CFPB['Consumer complaint narrative'].isnull().sum()
pct_miss_narr = no_miss_narr/len(df_CFPB.index) * 100
print(f"Number of missing complaints without narratives: {no_miss_narr}")
print(f"Percentage of all complaints without narrative: {pct_miss_narr:.2f}%")

del no_miss_narr
del pct_miss_narr
gc.collect()


# In[5]:


# Splitting the dataset into one with narratives and one without
df_CFPB_no_narr = df_CFPB[df_CFPB['Consumer complaint narrative'].isnull()].drop('Consumer complaint narrative', axis=1)
df_CFPB_w_narr = df_CFPB[df_CFPB['Consumer complaint narrative'].notna()]
print(df_CFPB_no_narr.shape)
print(df_CFPB_w_narr.shape)

del df_CFPB
gc.collect()


# In[6]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
no_narr_top_prods = df_CFPB_no_narr['Product'].value_counts(normalize=True).sort_values(ascending=False)[:6]
w_narr_top_prods = df_CFPB_w_narr['Product'].value_counts(normalize=True).sort_values(ascending=False)[:6]
no_narr_top_prods.plot(kind='barh', ax=axes[0])
w_narr_top_prods.plot(kind='barh', ax=axes[1])
axes[0].title.set_text('Product share of complaints with no narrative')
axes[1].title.set_text('Product share of complaints with narratives')
fig.subplots_adjust(hspace=.4)


# In[7]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
no_narr_top_prods = df_CFPB_no_narr['Company'].value_counts(normalize=True).sort_values(ascending=False)[:10]
w_narr_top_prods = df_CFPB_w_narr['Company'].value_counts(normalize=True).sort_values(ascending=False)[:10]
no_narr_top_prods.plot(kind='barh', ax=axes[0])
w_narr_top_prods.plot(kind='barh', ax=axes[1])
axes[0].title.set_text('Company share of complaints with no narrative')
axes[1].title.set_text('Company share of complaints with narratives')
fig.subplots_adjust(hspace=.4)


# In[8]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
no_narr_top_prods = df_CFPB_no_narr['State'].value_counts(normalize=True).sort_values(ascending=False)[:10]
w_narr_top_prods = df_CFPB_w_narr['State'].value_counts(normalize=True).sort_values(ascending=False)[:10]
no_narr_top_prods.plot(kind='barh', ax=axes[0])
w_narr_top_prods.plot(kind='barh', ax=axes[1])
axes[0].title.set_text('State share of complaints with no narrative')
axes[1].title.set_text('State share of complaints with narratives')
fig.subplots_adjust(hspace=.4)


# # Sentiment Analysis

# In[9]:


from nrclex import NRCLex 
import glob
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()


# In[10]:


df_narrative = pd.read_csv("narratives.csv") # tokenized data
df_narrative.head()


# In[11]:


EQUIFAX_w_narr=df_narrative[df_narrative['Company'].isin(['EQUIFAX, INC.'])==True]
content=EQUIFAX_w_narr['Consumer complaint narrative']
EQUIFAX_w_narr_subset=pd.DataFrame({'narrative': content})
EQUIFAX_w_narr_subset.reset_index(drop=True, inplace=True)
EQUIFAX_w_narr_subset=EQUIFAX_w_narr_subset[0:200]
EQUIFAX_w_narr_subset.head()


# # Sentiment Analysis w/ EmoLex and TfidfVectorizer

# In[12]:


filepath = "NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt"
emolex_df = pd.read_csv(filepath,  names=["word", "emotion", "association"], skiprows=45, sep='\t', keep_default_na=False)
emolex_df = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
emolex_df.head()


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(vocabulary=emolex_df.word,
                      use_idf=False, 
                      norm='l1')


# In[14]:


matrix = vec.fit_transform(EQUIFAX_w_narr_subset.narrative)


# In[15]:


vocab = vec.get_feature_names()


# In[16]:


wordcount_df = pd.DataFrame(matrix.toarray(), columns=vocab)


# In[17]:


wordcount_df.head()


# In[18]:


#wordcount_df.sort_values(by='credit', ascending=False).head(5)


# In[19]:


emolex_df.head()


# In[20]:


emolex_df[emolex_df.anger == 1].head()


# In[21]:


neg_words = emolex_df[emolex_df.negative == 1]['word']
neg_words.head(10)


# In[22]:


wordcount_df[neg_words].head()


# In[23]:


negative_words = emolex_df[emolex_df.negative == 1].word

EQUIFAX_w_narr_subset['negative'] = wordcount_df[negative_words].sum(axis=1)
EQUIFAX_w_narr_subset.head(10)


# In[24]:


positive_words = emolex_df[emolex_df.positive == 1].word

EQUIFAX_w_narr_subset['positive'] = wordcount_df[positive_words].sum(axis=1)
EQUIFAX_w_narr_subset.head(10)


# In[25]:


angry_words = emolex_df[emolex_df.anger == 1]['word']
EQUIFAX_w_narr_subset['anger'] = wordcount_df[angry_words].sum(axis=1)
EQUIFAX_w_narr_subset.head(10)


# In[26]:


pd.options.mode.chained_assignment = None 


# In[27]:


EQUIFAX_w_narr_subset.plot(x='positive', y='negative', kind='scatter')


# In[28]:


EQUIFAX_w_narr_subset.plot(x='positive', y='anger', kind='scatter')


# In[29]:


#EQUIFAX_w_narr['product'] = wordcount_df[['credit', 'debt', 'account', 'mortgage']].sum(axis=1)
#EQUIFAX_w_narr.head(10)


# In[30]:


#wordcount_df[['awful', 'dispute', 'bad', 'worse', 'incorrect']].head()


# # Sentiment Analysis before Tokenization with TextBlob
# used data: all equifax data w/ narrative from stm_labeled_data

# In[31]:


df = pd.read_csv('stm_labeled_data',index_col=0)
df.reset_index(drop=True, inplace=True)
df.head()


# In[32]:


list(df.columns)


# In[33]:


EQUIFAX_w_narr=df[df['Company'].isin(['EQUIFAX, INC.'])==True]
#content=EQUIFAX_w_narr['Consumer.complaint.narrative']

#import random
#random.sample(list(content), 500)

#EQUIFAX_w_narr_subset=pd.DataFrame({'narrative': content})
#EQUIFAX_w_narr_subset.reset_index(drop=True, inplace=True)
#EQUIFAX_w_narr_subset.head()
EQUIFAX_w_narr.reset_index(drop=True, inplace=True)
EQUIFAX_w_narr


# In[34]:


import re
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


# In[35]:


print(type(EQUIFAX_w_narr_subset['narrative']))


# In[36]:


EQUIFAX_w_narr['sentiment'] = EQUIFAX_w_narr['Consumer.complaint.narrative'].apply(lambda narrative: TextBlob(narrative).sentiment)


# In[37]:


EQUIFAX_w_narr['polarity'] = EQUIFAX_w_narr['sentiment'].apply(lambda sentiment: sentiment[0])
EQUIFAX_w_narr['subjectivity'] = EQUIFAX_w_narr['sentiment'].apply(lambda sentiment: sentiment[1])

EQUIFAX_w_narr.head()


# In[38]:


EQUIFAX_w_narr['positive'] = EQUIFAX_w_narr['polarity'].apply(lambda polarity: polarity>0)
EQUIFAX_w_narr['neutral'] = EQUIFAX_w_narr['polarity'].apply(lambda polarity: polarity==0)
EQUIFAX_w_narr['negative'] = EQUIFAX_w_narr['polarity'].apply(lambda polarity: polarity<0)

EQUIFAX_w_narr.head()


# In[39]:


print("Positive comments number: {}".format(EQUIFAX_w_narr['positive'].sum()))
print("Positive comments percentage: {} %".format(100*EQUIFAX_w_narr['positive'].sum()/EQUIFAX_w_narr['Consumer.complaint.narrative'].count()))
print("Neutral comments number: {}".format(EQUIFAX_w_narr['neutral'].sum()))
print("Neutral comments percentage: {} %".format(100*EQUIFAX_w_narr['neutral'].sum()/EQUIFAX_w_narr['Consumer.complaint.narrative'].count()))
print("Negative comments number: {}".format(EQUIFAX_w_narr['negative'].sum()))
print("Negative comments percentage: {} %".format(100*EQUIFAX_w_narr['negative'].sum()/EQUIFAX_w_narr['Consumer.complaint.narrative'].count()))


# In[40]:


plt.figure(figsize=(50,30))
plt.margins(0.02)
plt.xlabel('Sentiment', fontsize=50)
plt.xticks(fontsize=40)
plt.ylabel('Frequency', fontsize=50)
plt.yticks(fontsize=40)
plt.hist(EQUIFAX_w_narr['polarity'], bins=50)
plt.title('Sentiment Distribution', fontsize=60)
plt.show()


# In[41]:


polarity_avg = EQUIFAX_w_narr.groupby('Product')['polarity'].mean().plot(kind='bar', figsize=(50,30))
plt.xlabel('Product', fontsize=45)
plt.ylabel('Average Sentiment', fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Average Sentiment per Product', fontsize=50)
plt.show()


# In[42]:


letter_avg = EQUIFAX_w_narr.groupby('Product')['Complaint.length'].mean().plot(kind='bar', figsize=(50,30))
plt.xlabel('Product', fontsize=35)
plt.ylabel('Count of Letters in Complaint', fontsize=35)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.title('Average Number of Letters per Rating Distribution', fontsize=40)
plt.show()


# In[43]:


correlation = EQUIFAX_w_narr[['polarity','Complaint.length']].corr()
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(50,30))
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
sns.heatmap(correlation, cmap='coolwarm', annot=True, annot_kws={"size": 40}, linewidths=10, vmin=-1.5, mask=mask)


# ### API

# In[44]:


import requests
import json


# In[45]:


class CFPB_Client(object): 
    ''' 
    Generic CFPB Class for sentiment analysis. 
    '''
    def __init__(self): 
        ''' 
        Class constructor or initialization method. 
        '''
        # keys and tokens from the Twitter Dev Console 
        consumer_key = 'XXXXXXXXXXXXXXXXXXXXXXXX'
        consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        access_token_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXX'
  
        # attempt authentication 
        try: 
            # create OAuthHandler object 
            self.auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            self.auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            self.api = tweepy.API(self.auth) 
        except: 
            print("Error: Authentication Failed") 
  
    def clean_tweet(self, tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) 
                                    |(\w+:\/\/\S+)", " ", tweet).split()) 
  
    def get_tweet_sentiment(self, tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(self.clean_tweet(tweet)) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'
  
    def get_tweets(self, query, count = 10): 
        ''' 
        Main function to fetch tweets and parse them. 
        '''
        # empty list to store parsed tweets 
        tweets = [] 
  
        try: 
            # call twitter api to fetch tweets 
            fetched_tweets = self.api.search(q = query, count = count) 
  
            # parsing tweets one by one 
            for tweet in fetched_tweets: 
                # empty dictionary to store required params of a tweet 
                parsed_tweet = {} 
  
                # saving text of tweet 
                parsed_tweet['text'] = tweet.text 
                # saving sentiment of tweet 
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 
  
                # appending parsed tweet to tweets list 
                if tweet.retweet_count > 0: 
                    # if tweet has retweets, ensure that it is appended only once 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet) 
  
            # return parsed tweets 
            return tweets 
  
        except tweepy.TweepError as e: 
            # print error (if any) 
            print("Error : " + str(e)) 
  
def main(): 
    # creating object of TwitterClient Class 
    api = TwitterClient() 
    # calling function to get tweets 
    tweets = api.get_tweets(query = 'Donald Trump', count = 200) 
  
    # picking positive tweets from tweets 
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
    # percentage of positive tweets 
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
    # picking negative tweets from tweets 
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
    # percentage of negative tweets 
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
    # percentage of neutral tweets 
    print("Neutral tweets percentage: {} % \ 
        ".format(100*(len(tweets) -(len( ntweets )+len( ptweets)))/len(tweets))) 
  
    # printing first 5 positive tweets 
    print("\n\nPositive tweets:") 
    for tweet in ptweets[:10]: 
        print(tweet['text']) 
  
    # printing first 5 negative tweets 
    print("\n\nNegative tweets:") 
    for tweet in ntweets[:10]: 
        print(tweet['text']) 
  
if __name__ == "__main__": 
    # calling main function 
    main() 


# In[ ]:


def get_sentiment()

