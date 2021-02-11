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
import contractions
import sys
import itertools
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm.notebook import tqdm


# # **Comparison between complaints with and without narratives**

# In[2]:


df_CFPB = pd.read_csv("data/complaints.csv")


# In[46]:


df_CFPB.head()


# In[97]:


df_CFPB.drop(columns=['Complaint ID', 'ZIP code'], inplace=True)


# In[98]:


for col in ['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company public response', 'State', 'Tags', 'Consumer consent provided?', 'Submitted via', 'Company response to consumer', 'Timely response?', 'Consumer disputed?']:
    df_CFPB[col] = df_CFPB[col].astype('category')


# In[49]:


df_CFPB.info(verbose=True)


# In[3]:


# Number of missing complaint narratives
no_miss_narr = df_CFPB['Consumer complaint narrative'].isnull().sum()
pct_miss_narr = no_miss_narr/len(df_CFPB.index) * 100
print(f"Number of missing complaints without narratives: {no_miss_narr}")
print(f"Percentage of all complaints without narrative: {pct_miss_narr:.2f}%")

del no_miss_narr
del pct_miss_narr
gc.collect()


# In[4]:


# Splitting the dataset into one with narratives and one without
df_CFPB_no_narr = df_CFPB[df_CFPB['Consumer complaint narrative'].isnull()].drop('Consumer complaint narrative', axis=1)
df_CFPB_w_narr = df_CFPB[df_CFPB['Consumer complaint narrative'].notna()]
print(df_CFPB_no_narr.shape)
print(df_CFPB_w_narr.shape)

del df_CFPB
gc.collect()


# In[52]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
no_narr_top_prods = df_CFPB_no_narr['Product'].value_counts(normalize=True).sort_values(ascending=False)[:6]
w_narr_top_prods = df_CFPB_w_narr['Product'].value_counts(normalize=True).sort_values(ascending=False)[:6]
no_narr_top_prods.plot(kind='barh', ax=axes[0])
w_narr_top_prods.plot(kind='barh', ax=axes[1])
axes[0].title.set_text('Product share of complaints with no narrative')
axes[1].title.set_text('Product share of complaints with narratives')
fig.subplots_adjust(hspace=.4)

for ax in axes.flatten():
    ax.invert_yaxis()


# In[53]:


fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
fig.set_size_inches(8, 4)
fig.tight_layout()

no_narr_3_prods = no_narr_top_prods[:3].sort_index()
w_narr_3_prods = w_narr_top_prods[:3].sort_index()

for i in range(3):
    df_CFPB_no_narr[df_CFPB_no_narr.Product == no_narr_3_prods.index[i]]['Sub-product'].value_counts(normalize=True).sort_values(ascending=False)[:3].plot(kind='barh', ax=axes[i,0])
    df_CFPB_w_narr[df_CFPB_w_narr.Product == w_narr_3_prods.index[i]]['Sub-product'].value_counts(normalize=True).sort_values(ascending=False)[:3].plot(kind='barh', ax=axes[i,1])

for i in range(3):
    axes[i, 1].tick_params(left=False, labelleft=False, labelright=True)
    axes[i,0].title.set_text(no_narr_3_prods.index[i])
    axes[i,1].title.set_text(w_narr_3_prods.index[i])
    
axes[0,0].title.set_text('Credit report, repair...')
axes[0,1].title.set_text('Credit report, repair...')

for ax in axes.flatten():
    ax.invert_yaxis()

fig.subplots_adjust(hspace=.4)
plt.show()


# In[54]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
no_narr_top_issues = df_CFPB_no_narr['Issue'].value_counts(normalize=True).sort_values(ascending=False)[:6]
w_narr_top_issues = df_CFPB_w_narr['Issue'].value_counts(normalize=True).sort_values(ascending=False)[:6]
no_narr_top_issues.plot(kind='barh', ax=axes[0])
w_narr_top_issues.plot(kind='barh', ax=axes[1])
axes[0].title.set_text('Issue share of complaints with no narrative')
axes[1].title.set_text('Issue share of complaints with narratives')
fig.subplots_adjust(hspace=.4)

for ax in axes.flatten():
    ax.invert_yaxis()


# In[55]:


fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
fig.set_size_inches(8, 4)
fig.tight_layout()

no_narr_4_issues = no_narr_top_issues[no_narr_top_issues.index.isin(w_narr_top_issues.index)].sort_index()
w_narr_4_issues = w_narr_top_issues[w_narr_top_issues.index.isin(no_narr_top_issues.index)].sort_index()

# Due to lack of data
no_narr_4_issues.drop(index='Incorrect information on credit report', inplace=True)
w_narr_4_issues.drop(index='Incorrect information on credit report', inplace=True)

for i in range(3):
    df_CFPB_no_narr[df_CFPB_no_narr.Issue == no_narr_4_issues.index[i]]['Sub-product'].value_counts(normalize=True).sort_values(ascending=False)[:3].plot(kind='barh', ax=axes[i,0])
    df_CFPB_w_narr[df_CFPB_w_narr.Issue == w_narr_4_issues.index[i]]['Sub-product'].value_counts(normalize=True).sort_values(ascending=False)[:3].plot(kind='barh', ax=axes[i,1])

for i in range(3):
    axes[i, 1].tick_params(left=False, labelleft=False, labelright=True)
    axes[i,0].title.set_text(no_narr_4_issues.index[i])
    axes[i,1].title.set_text(w_narr_4_issues.index[i])
    
axes[2,0].title.set_text('Problem with investigation...')
axes[2,1].title.set_text('Problem with investigation...')

for ax in axes.flatten():
    ax.invert_yaxis()

fig.subplots_adjust(hspace=.6)
plt.show()


# In[56]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
no_narr_top_issues = df_CFPB_no_narr['Company public response'].value_counts(normalize=True).sort_values(ascending=False)[:6]
w_narr_top_issues = df_CFPB_w_narr['Company public response'].value_counts(normalize=True).sort_values(ascending=False)[:6]
no_narr_top_issues.plot(kind='barh', ax=axes[0])
w_narr_top_issues.plot(kind='barh', ax=axes[1])
axes[0].title.set_text('Company public response share of complaints with no narrative')
axes[1].title.set_text('Company public response share of complaints with narratives')
fig.subplots_adjust(hspace=.4)

for ax in axes.flatten():
    ax.invert_yaxis()


# In[57]:


fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
no_narr_top_issues = df_CFPB_no_narr['Company response to consumer'].value_counts(normalize=True).sort_values(ascending=False)[:6]
w_narr_top_issues = df_CFPB_w_narr['Company response to consumer'].value_counts(normalize=True).sort_values(ascending=False)[:6]
no_narr_top_issues.plot(kind='barh', ax=axes[0])
w_narr_top_issues.plot(kind='barh', ax=axes[1])
axes[0].title.set_text('Company response to consumer share of complaints with no narrative')
axes[1].title.set_text('Company response to consumer share of complaints with narratives')
fig.subplots_adjust(hspace=.4)

for ax in axes.flatten():
    ax.invert_yaxis()


# In[5]:


df_CFPB_w_narr.to_csv('data/narratives.csv')
del df_CFPB_w_narr
del df_CFPB_no_narr
gc.collect()


# # **Analysing complaints with narratives**

# In[6]:


get_ipython().system('jupyter nbextension enable --py widgetsnbextension')
get_ipython().system('jupyter labextension install @jupyter-widgets/jupyterlab-manager')
get_ipython().run_line_magic('load_ext', 'memory_profiler')


# ### **Initial data exploration**

# In[7]:


df_narr = pd.read_csv('data/narratives.csv', converters={'Consumer complaint narrative': eval}, index_col=0)
df_narr.head()


# ##### **Average complaint length in words**

# In[8]:


get_ipython().run_line_magic('memit', '')
lengths = pd.Series(dtype='Int8')
narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)
for chunk in tqdm(narr_file_read):
    lengths = lengths.append(chunk['Consumer complaint narrative'].str.split().swifter.progress_bar(False).apply(len))


# In[9]:


lengths.sample(5)


# In[10]:


get_ipython().run_line_magic('memit', '')
fig, ax = plt.subplots()
ax.set_xlim(0,1500)
ax.hist(lengths, bins=100);


# In[11]:


avg = np.mean(lengths)
med = np.median(lengths)
print(f"The average number of words in a complaint is {round(avg, 2)}")
print(f"The median number of words in a complaint is {med}")


# In[12]:


df_narr['Complaint length'] = lengths.values
df_narr.head()


# In[13]:


# Checkpoint
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.csv')


# In[14]:


del lengths
gc.collect()


# #### **Cleaning the text data**

# ##### **Dropping "one word" complaints**

# In[15]:


df_narr['Complaint length'].describe()


# In[16]:


df_narr[df_narr['Complaint length']==1].sample()


# In[17]:


len(df_narr[df_narr['Complaint length']==1])


# In[18]:


get_ipython().run_line_magic('memit', '')
# Since 106 is insiginificant given the size of the dataset, we choose to drop it
df_narr.drop(df_narr[df_narr['Complaint length']==1].index, inplace = True)
print(df_narr.shape)


# In[19]:


len(df_narr[df_narr['Complaint length']==1])


# In[20]:


# Checkpoint
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.csv')
gc.collect()


# ##### **Expanding contractions**

# In[21]:


get_ipython().run_line_magic('memit', '')
narratives = pd.Series(dtype='string')
narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)

for chunk in tqdm(narr_file_read):
    narratives = narratives.append(pd.Series([' '.join(map(str, l)) for l in chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(lambda x: [contractions.fix(word) for word in x.split()])]), ignore_index=True)


# Sanity check

# In[22]:


df_narr.reset_index(drop=True, inplace=True)
df_narr['Consumer complaint narrative'][df_narr['Consumer complaint narrative'].str.contains("can't")].head(3)


# In[23]:


df_narr['Consumer complaint narrative'][df_narr['Consumer complaint narrative'].str.contains("can't")].iloc[0]


# In[24]:


narratives.iloc[53]


# Note that the can't in the last line has been changed to can not after processing

# In[25]:


searchfor = ["can't", "won't", "couldn't", "shouldn't", "I've", "wouldn't", "would've"]
narratives[narratives.str.contains('|'.join(searchfor))]


# We find that there remains erroneous entries (with sentences combined into long, incomprehensible words). We remove these from the dataset. 

# In[26]:


error_ids = [880, 15663, 27281, 28502, 49628, 55038, 56525, 64020, 70259, 369628]
narratives.drop(error_ids, inplace=True)
df_narr.drop(error_ids, inplace=True)
print(f"{len(error_ids)} rows dropped")
print(f"{len(df_narr)} rows remaining")


# In[27]:


df_narr['Consumer complaint narrative'] = narratives.values


# In[28]:


df_narr.reset_index(drop=True, inplace=True)


# In[29]:


df_narr.astype({"Complaint length": 'uint8'})


# In[30]:


df_narr.tail(1)


# In[31]:


# Checkpoint
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.csv')


# In[32]:


del narratives
del error_ids
gc.collect()


# ##### **Cleaning text (removing punctuation, lowercasing)**

# In[46]:


def clean_text(str_in):
    import re
    tmp = re.sub("[^A-z]+", " ", str_in.lower())
    return tmp


# In[47]:


get_ipython().run_line_magic('memit', '')
cleaned = pd.Series(dtype='string')
narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)

for chunk in tqdm(narr_file_read):
    cleaned = cleaned.append(chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(clean_text))


# In[51]:


for sentence in cleaned.values:
    print(type(sentence))
    print(sentence)
    break


# In[52]:


df_narr['Consumer complaint narrative'] = cleaned.values
df_narr.head(3)


# In[53]:


# Checkpoint
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.csv')


# In[54]:


del cleaned
gc.collect()


# ##### **Tokenization**

# In[58]:


get_ipython().run_line_magic('memit', '')
narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)
tokenized = pd.Series(dtype='string')

for chunk in tqdm(narr_file_read):
    tokenized = tokenized.append(chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(word_tokenize))


# In[59]:


for sentence in tokenized.values:
    print(type(sentence))
    print(sentence)
    break


# In[60]:


df_narr['Consumer complaint narrative'] = tokenized.values


# In[61]:


# Checkpoint
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.csv')


# In[62]:


del tokenized
del chunk
gc.collect()


# ##### **POS tagging**

# In[2]:


narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)
pos_tagged = pd.Series(dtype='string')
nltk.download('averaged_perceptron_tagger')

for chunk in tqdm(narr_file_read):
    pos_tagged = pos_tagged.append(chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(nltk.tag.pos_tag))


# In[5]:


for sentence in pos_tagged.values:
    print(sentence)
    break


# In[6]:


df_narr['Consumer complaint narrative'] = pos_tagged.values


# In[7]:


# Checkpoint
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.csv')


# In[8]:


del pos_tagged
del chunk
gc.collect()


# ##### **Lemmatization**

# In[10]:


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[11]:


narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=5000, iterator=True)
lemmatized = pd.Series(dtype='string')
nltk.download('averaged_perceptron_tagger')

for chunk in tqdm(narr_file_read):
    lemmatized = lemmatized.append(chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(lambda x: [WordNetLemmatizer().lemmatize(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x]))


# In[13]:


for sentence in lemmatized.values:
    print(type(sentence))
    print(sentence)
    break


# In[14]:


df_narr['Consumer complaint narrative'] = lemmatized.values


# In[15]:


# Checkpoint
print('Saved to data/narratives.csv')
df_narr.to_csv('data/narratives.csv')


# In[16]:


del lemmatized
del chunk
gc.collect()


# ##### **Removing stopwords**

# In[17]:


narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)
stop_rem = pd.Series(dtype='string')
stop_words = set(stopwords.words('english'))

for chunk in tqdm(narr_file_read):
    stop_rem = stop_rem.append(chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(lambda x: [word for word in x if word not in stop_words]))


# In[18]:


for sentence in stop_rem.values:
    print(type(sentence))
    print(sentence)
    break


# In[19]:


df_narr['Consumer complaint narrative'] = stop_rem.values


# In[20]:


# Checkpoint
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.csv')


# ##### **Identifying the most common words**

# In[22]:


word_counter = collections.defaultdict(int)

for token in tqdm(itertools.chain(*df_narr['Consumer complaint narrative'])):
    word_counter[token] += 1


# In[23]:


word_counter = collections.Counter(word_counter)


# In[26]:


x, y = [], []
for word, count in word_counter.most_common(25):
    x.append(word)
    y.append(count)

sns.barplot(x=y, y=x)


# We observe the prevelance of unmeaningful tokens such as xxxx and xx

# ##### **Removing these odd tokens**

# In[27]:


odd_tokens = ['xxxx', 'xx']


# In[28]:


narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)
clean_tokens = pd.Series(dtype='string')

for chunk in tqdm(narr_file_read):
    clean_tokens = clean_tokens.append(chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(lambda x: [word for word in x if word not in odd_tokens]))


# In[29]:


df_narr['Consumer complaint narrative'] = clean_tokens.values


# In[30]:


word_counter = collections.defaultdict(int)
for token in tqdm(itertools.chain(*df_narr['Consumer complaint narrative'])):
    word_counter[token] += 1


# In[31]:


word_counter = collections.Counter(word_counter)


# In[ ]:


x, y = [], []
for word, count in word_counter.most_common(25):
    x.append(word)
    y.append(count)

sns.barplot(x=y, y=x)


# The odd tokens have been removed

# In[33]:


# Checkpoint
df_narr.to_pickle('data/narratives.pkl')
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.pkl')
print('Saved to data/narratives.csv')


# ##### **OPTIONAL: Combining list of tokens into a string**

# Some packages used for topic modeling/n-gram creation require that text data is fed in as a string, rather than a list of tokens. Run the following cells to perform this conversion if necessary. The cleaned data file in Github comes without this conversion.

# In[ ]:


# Reading in the data file if df_narr has not already been read in already
df_narr = pd.read_csv('data/narratives.csv', converters={'Consumer complaint narrative': eval}, chunksize=10000)


# In[ ]:


narr_file_read = pd.read_csv('data/narratives.csv', usecols=['Consumer complaint narrative'], converters={'Consumer complaint narrative': eval}, chunksize=10000, iterator=True)
stringed = pd.Series(dtype='string')

for chunk in tqdm(narr_file_read):
    stringed = stringed.append(chunk['Consumer complaint narrative'].swifter.progress_bar(False).apply(lambda x: [' '.join(x)]))


# In[8]:


stringed = pd.Series([string[0] for string in stringed])


# In[9]:


for sentence in stringed.values:
    print(sentence)
    print(type(sentence))
    break


# In[10]:


df_narr['Consumer complaint narrative'] = stringed.values


# In[ ]:


# Checkpoint
df_narr.to_pickle('data/narratives.pkl')
df_narr.to_csv('data/narratives.csv')
print('Saved to data/narratives.pkl')
print('Saved to data/narratives.csv')


# In[11]:


del stringed
del chunk
gc.collect()

