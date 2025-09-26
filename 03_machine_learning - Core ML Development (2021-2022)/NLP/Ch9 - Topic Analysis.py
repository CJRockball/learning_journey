#%% 9.1 Code to initialize the newsgroup training and test subsets

from sklearn.datasets import fetch_20newsgroups
import numpy as np

def load_dataset(a_set, cats):
    dataset = fetch_20newsgroups(subset=a_set, categories=cats, remove=('headers', 'footers', 'quotes'),
                                shuffle=True)
    return dataset

categories = ['comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
              'rec.sport.hockey','sci.crypt','sci.med','sci.space','talk.politics.mideast']

newsgroups_train = load_dataset('train', categories)
newsgroups_test = load_dataset('test', categories)

# %% 9.2 Code to run some general checks on the uploaded data

def check_data(dataset):
    print(list(dataset.target_names))
    print(dataset.filenames.shape)
    print(dataset.target.shape)
    if dataset.filenames.shape[0] == dataset.target.shape[0]:
        print('Equal sizes for data and targets')
    print(dataset.filenames[0])
    print(dataset.data[0])
    print(dataset.target[:10])

check_data(newsgroups_train)
print('\n***\n')
check_data(newsgroups_test)

#%% 9.3 Code to apply TfidfVectorize and convert texts into vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

def test2vec(vectorizer, train_set, test_set):
    vectors_train = vectorizer.fit_transform(train_set.data)
    vectors_test = vectorizer.transform(test_set.data)
    return vectors_train, vectors_test

vectors_train, vectors_test = test2vec(vectorizer, newsgroups_train, newsgroups_test)

print(vectors_train.shape)
print(vectors_test.shape)
print(vectors_train[0])
print(vectorizer.get_feature_names()[33404])

#%% 9.4 Code to perform topic classification with the Naive Bayes classifier

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=0.1)
clf.fit(vectors_train, newsgroups_train.target)
predictions = clf.predict(vectors_test)

# %% 9.5 Code to evaluate the results for this approach
from sklearn import metrics

def show_top(classifier, categories, vectorizer, n):
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    for i, category in enumerate(categories):
        cat_features = classifier.feature_log_prob_[1][i]
        top = np.argsort(cat_features)[-n:]
        print(f'{category}: {" ".join(feature_names[top])}')

full_report = metrics.classification_report(newsgroups_test.target, 
                                            predictions,
                                            target_names=newsgroups_test.target_names)
print(full_report)
show_top(clf, categories, vectorizer, 10)

# %% 9.6 Code to explore the classifier's error and confusions
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

classifier = clf.fit(vectors_train, newsgroups_train.target)

disp = ConfusionMatrixDisplay.from_estimator(classifier, vectors_test, newsgroups_test.target,
                              values_format='0.0f', cmap=plt.cm.Blues)

print(disp.confusion_matrix)

plt.show()
for i,category in enumerate(newsgroups_train.target_names):
    print(i,category)




# %%

import random
random.seed(42)

all_news = list(zip(newsgroups_train.data, newsgroups_train.target))
all_news = list(zip(newsgroups_test.data, newsgroups_test.target))

random.shuffle(all_news)

all_news_data = [text for (text,label) in all_news]
all_news_labels = [label for (text,label) in all_news]

print('Data:')
print(str(len(all_news_data)) + ' posts in ' + 
        str(np.unique(all_news_labels).shape[0]) + ' categories\n')

print('Labels: ')
print(all_news_labels[:10])
num_clusters = np.unique(all_news_labels).shape[0]
print('Actual number of clusters: ' + str(num_clusters))

# %% 9.8 Code to preprocess the data with TfidfVectorize and SVD
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, stop_words='english', use_idf=True)

def transform(data, vectorizer, dimensions):
    trans_data = vectorizer.fit_transform(data)
    print('Transformed data contains: ' + str(trans_data.shape[0]) +
          ' with ' + str(trans_data.shape[1]) + ' features =>')

    svd = TruncatedSVD(dimensions)
    pipe = make_pipeline(svd, Normalizer(copy=False)) 
    reduced_data = pipe.fit_transform(trans_data)
    return reduced_data, svd

reduced_data, svd = transform(all_news_data, vectorizer, 300)
print('Reduced data contains: ' + str(reduced_data.shape[0]) + " with " +
                                    str(reduced_data.shape[1]) + ' features')


#%% 9.9 Code to run the KMeans clustering algorithm
from sklearn.cluster import KMeans

def cluster(data, num_clusters):
    km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, random_state=0)
    km.fit(data)
    return km

km = cluster(reduced_data, num_clusters)

# %% 9.10 Code to evaluate the results obtained with the clustering algorithm

def evaluate(km, labels, svd):
    print('Clustering report \n')
    print(f'* Homogeneity: {str(metrics.homogeneity_score(labels, km.labels_))}')
    print(f'* Completness: {str(metrics.completeness_score(labels, km.labels_))}')
    print(f'* V-measure: {str(metrics.v_measure_score(labels, km.labels_))}')

    print('\nMost discriminative words per cluster:')
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:,::-1]

    terms = vectorizer.get_feature_names_out()
    for i in range(num_clusters):
        print('Cluster ' + str(i) + ": ")
        cl_terms =''
        for ind in order_centroids[i,:50]:
            cl_terms += terms[ind] + ' '
        print(cl_terms + '\n')

evaluate(km, all_news_labels, svd)

print('\nCategories:')
for i, categories in enumerate(newsgroups_train.target_names):
    print('*', category)

