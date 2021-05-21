# import packages.
import os
import pandas as pd
import numpy as np
import pysentiment2 as ps
from nltk.corpus import stopwords, PlaintextCorpusReader
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


# obtain the current directory.
os.getcwd()

# list of stopwords and sentiment score.
stop_words = set(stopwords.words('english'))
lm = ps.LM()

# load data.
textual = pd.read_csv("textual_data_vacc.csv")
textual.loc[textual['body'].isna(), 'body'] = 'None'
# combine title and body.
textual['text'] = textual['title'] + ' ' + textual['body']
textual['timestamp'] = pd.to_datetime(textual['timestamp'])
textual['date'] = textual['timestamp'].dt.year * 10000 + \
                  textual['timestamp'].dt.month * 100 + textual['timestamp'].dt.day
# combine same date texture.
df = pd.DataFrame(np.sort(textual['date']), columns=['date']).drop_duplicates()
df.reset_index(drop=True, inplace=True)
textual['text'] = textual['text'].apply(lambda x: ' ' + x)
text = textual.groupby(by='date')['text'].sum()
text.reset_index(drop=True, inplace=True)
df['text'] = text
score = textual.groupby(by='date')['score'].sum()
score.reset_index(drop=True, inplace=True)
df['score'] = score
# set lower case and remove non-English-letter.
df['text'] = df['text'].str.lower()
df['text'].replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# process the textual data.
head_clean = []
sent_score = []
for i in range(len(df)):
    # tokenized
    tokens = word_tokenize(df['text'][i])
    tokens = [w for w in tokens if w not in ['comment', 'none']]
    # stemmed
    stemmed = [PorterStemmer().stem(word) for word in tokens]
    # remove stopwords.
    words = [w for w in stemmed if w not in stop_words]
    head_clean.append(' '.join(word for word in words))
    sent_score.append(lm.get_score(tokens)['Polarity'])
df['Processed'] = head_clean
df['sent_score'] = sent_score

# save the Processed information as separated txt.
os.chdir("C:/Users/ddnd/Desktop/Current Work/mfe431ml_pre/corpus")
for ind in range(len(df)):
    file_id = str(df['date'][ind])
    with open('file_' + file_id + '.txt', 'w') as fout:
        fout.write(df['Processed'][ind])
        fout.close()

# create the corpus.
new_corpus = PlaintextCorpusReader(
    'C:/Users/ddnd/Desktop/Current Work/mfe431ml_pre/corpus', '.*')


# define the DTM function.
def dtm_from_corpus(x_corpus):
    s = 0
    fd_list = []
    for x in range(s, len(x_corpus.fileids())):
        fd_list.append(nltk.FreqDist(x_corpus.words(x_corpus.fileids()[x])))
    output = pd.DataFrame(fd_list, index=x_corpus.fileids()[s:])
    output.fillna(0, inplace=True)
    return output


# create DTM.
dtm = dtm_from_corpus(new_corpus)
freq = dtm.sum()
# for freq > 25.
subset = freq[freq > 25]
names = subset._stat_axis.values.tolist()
print(subset)
# plot the biggest 30.
ax1 = plt.figure()
sns.barplot(x=subset[0:30], y=names[0:30])
plt.show()
