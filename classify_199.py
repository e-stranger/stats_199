from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier

import pandas

def get_grouped_labels(label):
	"""Assumes labels `vneg`, `neg`, `neu`, `pos`, `vpos`"""
	if label == 'vneg':
		return 'neg'
	elif label == 'vpos':
		return 'pos'
	return label

if __name__ == "__main__":
	df = pandas.read_csv("data/articles_sample.csv")
	classified = df.loc[df['new?'].apply(lambda x: x == 1)]
	classified = classified.loc[classified['class_t'].apply(lambda x: x != "dm")]
	classified.to_csv("classfied.csv")
	labels = classified['class_t']
	grouped_labels = [get_grouped_labels(label) for label in labels]
	documents = classified['content']
	vec = TfidfVectorizer(max_features=1000, stop_words='english')
	doc_vectors = MinMaxScaler().fit_transform(vec.fit_transform(documents).toarray())
	clf = LogisticRegression()
	clf = MLPClassifier(hidden_layer_sizes=(1000,1000,1000))
	print(cross_val_score(clf, doc_vectors, grouped_labels))
	cvp = cross_val_predict(clf, doc_vectors, grouped_labels)
	print(confusion_matrix(grouped_labels, cvp , labels=["pos", "neu", "neg"]))
