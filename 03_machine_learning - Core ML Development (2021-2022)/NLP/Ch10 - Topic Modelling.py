#%% 10.1 Code to access the Newsgroups data on specific topics
from sklearn.datasets import fetch_20newsgroups

def load_dataset(sset, cats):
    if cats == []:
        newsgroups_dset = fetch_20newsgroups(subset=sset, 
                                             remove=('headers', 'footers','quotes'), shuffle=True)

    else: 
        newsgroups_dset = fetch_20newsgroups(subset=sset, categories=cats,
                                             remove=('headers', 'footers','quotes'), shuffle=True)
    return newsgroups_dset

categories = ['comp.windows.x', 'misc.forsale', 'rec.autos']
categories += ['rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
categories += ['sci.crypt', 'sci.med', 'sci.space']
categories += ['talk.politics.mideast']

newsgroups_all = load_dataset('all', categories)
print(len(newsgroups_all.data))

# %% 10.2 Code to preprocess the data using NLTK and gensim
import nltk
import gensim
from nltk.stem import SnowballStemmer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as stopwords

stemmer = SnowballStemmer('english')

def stem(text):
    return stemmer.stem(text)

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text, min_len=4):
        if token not in stopwords:
            result.append(stem(token))
    return result

#%% 10.3 Code to inspect the result of the preprocessing step

doc_sample = newsgroups_all.data[0]
print('Original document: ')
print(doc_sample)

print('\n\nTokenized document: ')
words = []
for token in gensim.utils.tokenize(doc_sample):
    words.append(token)
print(words)

print('\n\nPreprocessed document: ')
print(preprocess(doc_sample))

#10.4 Code to inspect the preprocessing output for a group of documents
for i in range(0,10):
    print(str(i) + '\t' + ', '.join(preprocess(newsgroups_all.data[i])[:10]))


# %% 10.5 Code to convert word content of the documents into a dictionary

processed_docs = []
for i in range(0, len(newsgroups_all.data)):
    processed_docs.append(preprocess(newsgroups_all.data[i]))

print(len(processed_docs))

dictionary = gensim.corpora.Dictionary(processed_docs)
print(len(dictionary))

index = 0
for key, value in dictionary.iteritems():
    print(key, value)
    index += 1
    if index > 9:
        break



# %% 10.6 Code to preform further dimensionality reduction on the documents

dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=10000)
print(len(dictionary))

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[0])


# %% 10.7 Code to check word stems behind IDs from the dictionary

bow_doc = bow_corpus[0]
for i in range(len(bow_doc)):
    print(f"Key {bow_doc[i][0]} =\"{dictionary[bow_doc[i][0]]}\":\
        occurences={bow_doc[i][1]}")

# %% 10.8 Code to run the LDA algorithm onn your document
id2word = dictionary
corpus = bow_corpus

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,
                                           num_topics=10, random_state=100,
                                           update_every=1, chunksize=1000,
                                           passes=10, alpha='symmetric',
                                           iterations=100, per_word_topics=True)

for index, topic in lda_model.print_topics(-1):
    print(f'Topic: {index} \n Words: {topic}')

# %% 10.9 Code to identify the main topic for each document in the collection

def analyse_topics(ldamodel, corpus, texts):
    main_topic = {}
    percentage = {}
    keywords = {}
    text_snippets = {}
    
    for i, topic_list in enumerate(ldamodel[corpus]):
        topic = topic_list[0]
        topic = sorted(topic, key=lambda x: (x[1]), reverse=True)
    
        for j, (topic_num, prop_topic) in enumerate(topic):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ', '.join([word for word, prop in wp[:5]])
                main_topic[i] = int(topic_num)
                percentage[i] = round(prop_topic,4)
                keywords[i] = topic_keywords
                text_snippets[i] = texts[i][:8]
            else:
                break
    return main_topic, percentage, keywords, text_snippets


main_topic, percentage, keywords, text_snippets = analyse_topics(
    lda_model, bow_corpus, processed_docs)

# %% 10.10 Code to print out the main topic for each document in the collection

indexes = []
rows = []
for i in range(0, 10):
    indexes.append(i)
rows.append(['ID', 'Main Topic', 'Contribution (%)', 'Keywords', 'Snippet'])

for idx in indexes:
    rows.append([str(idx), f'{main_topic.get(idx)}',
                 f'{percentage.get(idx):.4f}',
                 f'{keywords.get(idx)}\n',
                 f'{text_snippets.get(idx)}'])

columns = zip(*rows)
column_widths = [max(len(item) for item in col) for col in columns]
for row in rows:
    print(''.join(' {:{width}} '.format(row[i], width=column_widths[i]) for i in range(0, len(row))))

# %% 10.11 Code to visualize the output of LDA using pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
