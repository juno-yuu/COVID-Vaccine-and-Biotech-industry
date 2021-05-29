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
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk import ngrams
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import scikitplot as skplt
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from PIL import Image

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# obtain the current directory.
os.getcwd()

# list of stopwords and sentiment score.
stop_words = set(stopwords.words('english'))
lm = ps.LM()

# load data.
textual = pd.read_csv("textual_data_vacc.csv")
pfe = pd.read_csv('PFE.csv')
jnj = pd.read_csv('JNJ.csv')
mrna = pd.read_csv('MRNA.csv')
textual.loc[textual['body'].isna(), 'body'] = 'None'
# combine title and body.
textual['text'] = textual['title'] + ' ' + textual['body']
textual['timestamp'] = pd.to_datetime(textual['timestamp'])
textual['timestamp'] = textual['timestamp'] + pd.Timedelta(days=7)
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

# draft
pfe.loc[pfe['Adj Close'].shift(1) < pfe['Adj Close'], 'ind'] = 1
pfe.loc[pfe['Adj Close'].shift(1) >= pfe['Adj Close'], 'ind'] = 0
df['Date'] = pd.to_datetime(df['date'], format='%Y%m%d')
pfe['Date'] = pd.to_datetime(pfe['Date'])
reddit_pfe = pd.merge(df, pfe, on=['Date'], how='inner')
indicator = reddit_pfe['Date']
df = df.loc[df['Date'].isin(indicator)]
df.reset_index(drop=True, inplace=True)

# process the textual data.
head_clean = []
sent_score = []
additional_stopwords = ['comment', 'none', 'www', 'gov', 'https', 'com']
for i in range(len(df)):
    # tokenized
    tokens = word_tokenize(df['text'][i])
    tokens = [w for w in tokens if w not in additional_stopwords]
    # stemmed
    stemmed = [PorterStemmer().stem(word) for word in tokens]
    # remove stopwords.
    words = [w for w in stemmed if w not in stop_words]
    head_clean.append(' '.join(word for word in words))
    sent_score.append(lm.get_score(tokens)['Polarity'])
df['Processed'] = head_clean
df['sent_score'] = sent_score
entire_processed_text = ' '.join(doc for doc in head_clean)

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


def word_frequency(sentence):
    lemmatizer = WordNetLemmatizer()
    new_tokens = word_tokenize(sentence)
    new_tokens = [t for t in new_tokens if t.isalpha()]
    new_tokens = [lemmatizer.lemmatize(t) for t in new_tokens]

    # counts the words, pairs and trigrams
    counted = Counter(new_tokens)
    counted_2 = Counter(ngrams(new_tokens, 2))
    counted_3 = Counter(ngrams(new_tokens, 3))

    # create outputs.
    word1 = pd.DataFrame(counted.items(), columns=['word', 'frequency']).sort_values(by='frequency',
                                                                                     ascending=False)
    word2 = pd.DataFrame(counted_2.items(), columns=['pairs', 'frequency']).sort_values(by='frequency',
                                                                                        ascending=False)
    word3 = pd.DataFrame(counted_3.items(), columns=['trigrams', 'frequency']).sort_values(by='frequency',
                                                                                           ascending=False)
    return word1, word2, word3


data2, data3, data4 = word_frequency(entire_processed_text)

# Logistic Regression
x_data = dtm[1:]
y_data = reddit_pfe['ind'][1:]
x_data.index = y_data.index
# separate training/testing by ratio 4:1.
x_train = x_data[:64]  # until 03/24/21
x_test = x_data[64:]
y_train = y_data[:64]
y_test = y_data[64:]

mod1 = LogisticRegressionCV(penalty='elasticnet', solver='saga',
                            l1_ratios=[0.5], cv=5, max_iter=10000, fit_intercept=False, refit=True)
mod1 = mod1.fit(x_train, y_train)
value = pd.DataFrame(mod1.coef_)
value.columns = list(x_train)
ddnd = value.T
ddnd.reset_index(drop=False, inplace=True)
ddnd = ddnd.sort_values(0)
best_alpha_cv = mod1.C_

# Estimate the model again using this "best" alpha
mod1wp = LogisticRegression(penalty='elasticnet', solver='saga',
                            l1_ratio=0.5, max_iter=1000000000, fit_intercept=True)
mod1wp.C = best_alpha_cv[0]
mod1wp.fit(x_train, y_train)
valuewp = pd.DataFrame(mod1wp.coef_)
valuewp.columns = list(x_train)
ddndwp = valuewp.T
ddndwp.reset_index(drop=False, inplace=True)
ddndwp = ddndwp.sort_values(0)

ax = skplt.metrics.plot_roc(y_test, mod1wp.predict_proba(x_test), title='ROC Curves using Test Sample',
                            plot_micro=False, plot_macro=False, classes_to_plot=1,
                            ax=None, figsize=None, cmap='nipy_spectral',
                            title_fontsize="large", text_fontsize="medium")
plt.show()
prop = sum(np.where(mod1wp.predict(x_test) == y_test, 1, 0)) / len(y_test)
print("The proportion of right prediction is {:2.2%}".format(prop))

# sentiment test.
sent_words = ["efficaci", "cdc", "misinform", "vaccin", "covid",
              "immun", "infect", "autism", "risk", "unvaccin",
              "antibodi", "viru", "safe", "death", "medic", "diseas",
              "pfizer", "danger", "prevent", "spread", "inject", "protect", "die"]
x_sent_data = dtm[sent_words][1:]
x_sent_data.index = y_data.index
# separate training/testing by ratio 4:1.
x_sent_train = x_sent_data[:64]  # until 03/24/21
x_sent_test = x_sent_data[64:]
mod1_sent = LogisticRegressionCV(penalty='elasticnet', solver='saga',
                                 l1_ratios=[0.5], cv=5, max_iter=10000, fit_intercept=False, refit=True)
mod1_sent = mod1_sent.fit(x_sent_train, y_train)
value_sent = pd.DataFrame(mod1_sent.coef_)
value_sent.columns = list(x_sent_train)
ddnd_sent = value_sent.T
ddnd_sent.reset_index(drop=False, inplace=True)
ddnd_sent = ddnd_sent.sort_values(0)
best_alpha_cv_sent = mod1_sent.C_
# Estimate the model again using this "best" alpha
mod1wp_sent = LogisticRegression(penalty='elasticnet', solver='saga',
                                 l1_ratio=0.5, max_iter=1000000000, fit_intercept=True)
mod1wp_sent.C = best_alpha_cv_sent[0]
mod1wp_sent.fit(x_sent_train, y_train)
valuewp_sent = pd.DataFrame(mod1wp_sent.coef_)
valuewp_sent.columns = list(x_sent_train)
ddndwp_sent = valuewp_sent.T
ddndwp_sent.reset_index(drop=False, inplace=True)
ddndwp_sent = ddndwp_sent.sort_values(0)
ax_sent = skplt.metrics.plot_roc(y_test, mod1wp_sent.predict_proba(x_sent_test),
                                 title='ROC Curves (sent) using Test Sample',
                                 plot_micro=False, plot_macro=False, classes_to_plot=1,
                                 ax=None, figsize=None, cmap='nipy_spectral',
                                 title_fontsize="large", text_fontsize="medium")
plt.show()
prop_sent = sum(np.where(mod1wp_sent.predict(x_sent_test) == y_test, 1, 0)) / len(y_test)
print("The proportion of right prediction is {:2.2%}".format(prop_sent))
output = ddndwp_sent.iloc[[3, 4, 6, 7, 17, 20, 22]]


# word cloud.


def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = int(360 * 93 / 255.0)
    s = int(100 * 255.0 / 255.0)
    l = int(100 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)


mask = np.array(Image.open('C:/Users/ddnd/Desktop/Current Work/mfe431ml_pre/vacc.png'))
mask_colors = ImageColorGenerator(mask)
wc = WordCloud(max_words=1000, stopwords={'say', 'ha', 'wa', 'u', 'thi'},
               background_color="white", mask=mask,
               color_func=random_color_func).generate_from_text(entire_processed_text)
plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
