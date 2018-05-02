from flask import Flask, request, render_template
import pandas, re

app = Flask(__name__)

categories = []

opinion_url = 'http://thehill.com/opinion/'
whurl = 'http://thehill.com/opinion/white-house?page=2'
opinion_base = 'http://thehill.com{}'

subtitle = ''

df = pandas.read_csv("../data/articles_sample.csv")

def clean(text):
	text = re.sub("([\s]+)", u" ", text)
	text = re.sub("(^[\s])|([\s]$)", "", text)
	return text

def save_df():
	df.to_csv("../data/articles_sample.csv", index=False)

@app.after_request
def after_request(response):
	print("hello, world!")
	print(response)
	return response

@app.route("/", methods=["GET", "POST"])
def classify(previous=''):
	
	if request.method == "POST":
		id = int(request.form.get("article_id", None))

		lean_c = request.form.get("classification_c", None)
		lean_o = request.form.get("classification_o", None)
		lean_t = request.form.get("classification_t", None)
		edited_text = request.form.get("edited_text", None)

		c = 0
		t = 0
		o = 0

		if lean_c != "dm":
			c = 1
		if lean_t != "dm":
			t = 1
		if lean_o != "dm":
			o = 1
		print(id)
		print(lean_c)
		print(lean_o)
		print(lean_t)
		df.loc[id, "new?"] = 1
		df.loc[id,"class_o"] = lean_o
		df.loc[id,"class_c"] = lean_c
		df.loc[id,"class_t"] = lean_t
		df.loc[id,"t"] = t
		df.loc[id,"o"] = o
		df.loc[id,"c"] = c
		df.loc[id, "content"] = clean(edited_text)
		save_df()
		print(df.loc[id])


	if not previous:
		previous = "Please classify these by political lean, if you have the time :)"

	articles = df.loc[df['new?'].apply(lambda x: x == 0)].to_dict('records')

	if len(articles) == 0:
		df.to_csv("class_articles.csv")
		print("all done!")
		exit(0)

	article = articles[0]

	page_header = "{} articles out of {} left to classify".format(len(articles), df.shape[0])


	return render_template('classify.html', article=article, page_header = page_header)
	return return_articles(which="classify.html", page_header=previous, category='', articles=articles,
							   categories=categories, do_shorten=False)


def return_articles(which, page_header, category, articles, categories, do_shorten=True):
	mod_categories = [(topic_c, " ".join([c.capitalize() for c in topic_c.split("-")])) for topic_c in categories]
	if do_shorten:
		for article in articles:
			article["text"] = " ".join(article["text"].split(" ")[:200])
	if len(articles) > 20:
		articles = articles[:21]

	return render_template(which, page_header=page_header, category=category, articles=articles,
						   categories=mod_categories)

if __name__ == "__main__":
	app.run()
