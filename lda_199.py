from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from utils import get_logger, get_now
import pandas, sys, json, datetime, re, pickle
import numpy as np
# Latent Dirichlet Allocation with Scikit-Learn

sample_filename = 'data/articles_sample.csv'

logger = get_logger(__name__)
# from
# http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

def write_words_to_file(object, method, filename=None):
	object['method'] = method
	object['time'] = get_now()

	if not filename:
		filename = 'word_data/iter{}.json'.format(logger.get_iter())
	with open(filename, 'w') as file:
		file.write(json.dumps(object, indent=4))

def eval_lda(model, filename, n_topics, n_words, ngram_range, **kwargs):
	vec, lda_object = run_topic_modelling(model, filename=filename, n_topics=n_topics, ngram_range=ngram_range)
	results = return_top_words(lda_object, vec.get_feature_names(), n_words)
	write_words_to_file(results, model)

def return_top_words(model, feature_names, n_top_words):
	""" Saves top n_top_words from each topic, saves to json"""
	topic_object = {}
	topic_object['n_topics'] = len(model.components_)
	for topic_idx, topic in enumerate(model.components_):
		try:
			top_words = topic.argsort()[:-n_top_words - 1:-1]
		except IndexError:
			logger.exception("Asked for more words than existed in topic vocabulary")
			top_words = topic.argsort()
		topic_object[topic_idx] = [feature_names[i] for i in top_words.tolist()]
	return topic_object

def load_dataset(filename, sample=None):
	try:
		df = pandas.read_csv(filename)
		if sample:
			df = df.sample(sample)
		return df
	except FileNotFoundError:
		print("File not found!")
		return None
	except Exception as e:
		print(str(e))
		return None

def run_kmeans(filename, n_topics, ngram_range):
	data = load_dataset(filename)
	data.dropna(subset=['content', 'title'], inplace=True)
	vec = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_features=1500)
	titles = data['title'].apply(lambda x: re.sub("( - Breitbart)|([\s]+$)|(^[\s]+)|(as it happened)|(\u2019)|(\u2018)|(\u2013)|(\u201c)|(\u201d)|(\u00a0)|( - The New York Times)", " ", x))
	titles = titles.apply(lambda x: re.sub("[\s]+", u" ", x))
	transformed_docs = vec.fit_transform(titles)
	topic_model = KMeans(n_clusters=n_topics)
	clusters = topic_model.fit_predict(transformed_docs)
	clusters_labels = [{'index': index.item(), 'cluster_label': cluster_label.item(), 'title': title , 'publication': publication} for index,cluster_label,title, publication in zip(np.arange(0, clusters.shape[0]), clusters, titles, data['publication'])]
	cluster_df = pandas.DataFrame(clusters_labels)
	cluster_df.to_csv("clusters.csv")
	groups = {}
	for i in range(0, n_topics):
		groups[str(i)] = []
	for cl in clusters_labels:
		groups[str(cl['cluster_label'])].append((cl['index'], cl['title']))
	write_words_to_file(groups, 'kmeans')


# assumes content is in column named # `content`
def run_topic_modelling(model, filename, n_topics, ngram_range):
	data = load_dataset(filename)
	data.dropna(subset=['content', 'title'], inplace=True)
	vec = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_features=1000)
	transformed_docs = vec.fit_transform(data['title'])
	if model == "LDA":
		topic_model = LatentDirichletAllocation(n_components=n_topics, max_iter=100)
	else:
		topic_model = NMF(n_components=n_topics, max_iter=1000, random_state=1, solver="mu", beta_loss='kullback-leibler')
	topic_model.fit(transformed_docs)

	# Save pickled vectorizer, LDA object for later!
	# We will need this to assign labels later on.

	with open("pickle/lda_pickled.pkl", "wb") as file:
		pickle.dump(topic_model, file)
	with open("pickle/vec_pickled.pkl", "wb") as file:
		pickle.dump(vec, file)
	return vec, topic_model

def main(model, filename, n_topics, n_words, ngram_range):
	if model == "kmeans":
		run_kmeans(filename, n_topics, ngram_range)
	else:
		eval_lda(model, filename, n_topics, n_words, ngram_range)

# Want to get everything since inauguration: Jan 20th, 2017

def filter_data():
	filenames = ["article_data/articles1.csv", "article_data/articles2.csv", "article_data/articles3.csv"]
	df = pandas.DataFrame()
	for filename in filenames:
		df = df.append(pandas.read_csv(filename))


	inauguration = datetime.datetime(year=2017, day=20, month=1)
	df.date = pandas.to_datetime(df.date, yearfirst=True)
	df = df.loc[df.date > inauguration]
	df.to_csv("since_inaug.csv")

def convert_to_n_dim(filename, n_topics):
	try:
		df = pandas.read_csv(filename)
	except FileNotFoundError:
		logger.exception("File not found while attempting PCA!")
		exit(1)
	publications = pandas.unique(df['publication'])
	init_matrix = np.zeros((publications.shape[0], n_topics))

	index_converter = {}
	for i,publication in enumerate(publications):
		index_converter[publication] = i

	for i,j in enumerate(df.index):
		cluster = df.loc[j, 'cluster_label']
		init_matrix[index_converter[df.loc[j, 'publication']], cluster] += 1

	pca_model = PCA(n_components=2)

	return pca_model, init_matrix



if __name__ == "__main__":
	try:
		filename = 'article_data/{}'.format(sys.argv[1])
		model = sys.argv[2]
		if model != "LDA" and model != "NMF" and model != "kmeans":
			logger.error("Model must be either LDA or NMF! Exiting...")
			exit(1)
		n_topics = int(sys.argv[3])
		n_words = int(sys.argv[4])
		ngram_lower = int(sys.argv[5])
		ngram_upper = int(sys.argv[6])
		ngram_range = (ngram_lower, ngram_upper)
		main(model, filename, n_topics, n_words, ngram_range=ngram_range )

	except IndexError:
		print("Supply filename argument!")
		exit(1)

