from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas, sys

# Latent Dirichlet Allocation with Scikit-Learn

sample_filename = 'data/articles_sample.csv'

# from
# http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

s = """WASHINGTON  —     Donald J. Trump on Tuesday named as his chief trade negotiator a Washington lawyer who has long advocated protectionist policies, the latest sign that Mr. Trump intends to fulfill his campaign promise to get tough with China, Mexico and other trading partners. Mr. Trump also renewed his episodic campaign to persuade American companies to expand domestic manufacturing, criticizing General Motors via Twitter on Tuesday morning for making in Mexico some of the Chevrolet Cruze hatchbacks it sells domestically. Hours later, Mr. Trump claimed credit after Ford said it would expand vehicle production in Flat Rock, Mich. The choice of Robert Lighthizer (pronounced   ) to be the United States’ trade representative nearly completes Mr. Trump’s selection of top economic advisers and, taken together with the  ’s running commentary on Twitter, underscores Mr. Trump’s focus on making things in America. That is causing unease among some Republicans who regard Mr. Trump’s views on trade as dangerously retrograde, even as they embrace the bulk of his economic agenda.  Mainstream economists warn that protectionist policies like import taxes could impose higher prices on consumers and slow economic growth. But some Democrats are signaling a readiness to support Mr. Trump. Nine House Democrats held a news conference Tuesday with the A. F. L. . I. O. president, Richard Trumka, to urge renegotiation of the North American Free Trade Agreement with Mexico and Canada. “We wanted him to know that we’ll work with him on doing that,” Mr. Trumka said. “I don’t think he has enough Republican support to do it, and rewriting the rules of trade is a necessary first step in righting the economy for working people. ” Mr. Trump and his top advisers on trade, including Mr. Lighthizer, share a view that the United States in recent decades prioritized the ideal of free trade over its own  . They argue that other countries are undermining America’s industrial base by subsidizing their own export industries while impeding American importers. They regard this unfair competition as a key reason for the lackluster growth of the economy. In picking Mr. Lighthizer, who has spent much of the last few decades representing American steel producers in their frequent litigation of trade disputes, Mr. Trump is seeking to hire one of Washington’s top trade lawyers to enforce international trade agreements more vigorously. He must be confirmed by the Senate. “He will do an amazing job helping turn around the failed trade policies which have robbed so many Americans of prosperity,” Mr. Trump said in a statement. Mainstream Republicans have sought common ground with Mr. Trump, emphasizing, for example, the importance of enforcing trade rules, but they have not abandoned the party’s longtime advocacy for trade. Senator Orrin Hatch of Utah, the chairman of the Senate Finance Committee, which will hold hearings on Mr. Lighthizer’s nomination, issued a cautiously supportive statement Tuesday. “As the world and our economic competitors move to expand their global footprints, we can’t afford to be left behind in securing strong deals that will increase our access to new markets for   products and services,” Mr. Hatch said in a statement. “I look forward to a vigorous discussion of Bob’s trade philosophy and priorities. ” Mr. Trump has named a number of advisers on trade, leaving some ambiguity about the division of responsibilities. The   named the economist Peter Navarro, an outspoken critic of China, to lead a new White House office overseeing trade and industrial policy. Mr. Trump also said Wilbur Ross, the billionaire investor and choice for commerce secretary, will play a key role. Mr. Lighthizer, however, is the only member of the triumvirate with government experience. “Those who say U. S. T. R. will be subordinated to other agencies are mistaken,” said Alan Wolff, another former senior American trade official who was the steel industry’s   on trade with Mr. Lighthizer for nearly 20 years, citing Mr. Lighthizer’s encyclopedic knowledge of trade law. “He’ll be a dominant figure on trade, in harmony with Wilbur Ross and Navarro. ” There is also an ideological divide between the people Mr. Trump has named to oversee trade policy and his broader circle of advisers, which is populated by longstanding trade advocates like Gary D. Cohn, the president of Goldman Sachs, who will lead the National Economic Council Rex W. Tillerson, the chief executive of Exxon Mobil, tapped for secretary of state and Gov. Terry Branstad of Iowa, Mr. Trump’s choice for ambassador to China. Proponents of trade hope the broader circle, and congressional Republicans, will exert a moderating influence. “You’re seeing a pretty clear indication that there will be a focus on the enforcement of our trade agreements and on the letter of the law,” said Scott Lincicome, an international trade lawyer at White  Case. “But that doesn’t necessarily mean a significant turn toward protectionism. Even free trade guys like me support enforcement. ” Trade opponents on the left and the right, meanwhile, are hoping Mr. Trump means to break with several decades of   policy. “There’s going to be a war within the Trump administration on where they go with trade, and we’re hoping to energize the worker base he had to make sure they go in the right direction to benefit the American worker,” Mr. Trumka said. Mr. Trump’s promise to immediately designate China as a currency manipulator may offer an early test of the administration’s intentions. Economists see no evidence China is suppressing the value of its currency, although it has done so in the past. Mr. Lincicome said officially labeling China a currency manipulator despite the lack of recent evidence would signal that the administration is taking a hard line on trade issues. A broader shift in trade policy would unfold more slowly. Mr. Trump has promised to renegotiate Nafta the original process took most of three years. He has promised to pursue enforcement actions against other nations, but it takes time to mount cases. He has threatened to impose new tariffs on imports, but sweeping changes most likely would require congressional legislation. Mr. Trump already is seeking to exert influence by seizing the presidential bullhorn. “General Motors is sending Mexican made model of Chevy Cruze to U. S. car   free across border,” he wrote Tuesday on Twitter. “Make in U. S. A. or pay big border tax!” General Motors announced in 2015 that it would make the Cruze in Coahuila, Mexico. American manufacturers are moving   production to Mexico to take advantage of lower labor costs and because of declining domestic demand. They continue to build more expensive vehicles in the United States. Ford’s announcement Tuesday does not reverse that trend. The carmaker said it still planned to move production of the compact Ford Focus from Michigan to Mexico. But it said it would invest in a different Michigan plant to expand production of   vehicles, including its   pickup truck and the Mustang sports car, as well as a new   sport utility vehicle. “We are encouraged by the   policies that   Trump and the new Congress have indicated they will pursue,” said the company’s chief executive, Mark Fields. Mr. Lighthizer served as deputy United States trade representative in the Ronald Reagan administration, when he was involved in pressing Japan to reduce its restrictions on American imports and its subsidies for its own exports. Mr. Trump has criticized China for similar practices, setting the stage for a new round of confrontations. Reagan is often remembered as an advocate for free trade, but his administration in its early hours imposed a quota on Japanese auto imports. It was the first in a long series of measures aimed at putting pressure on the nation that was then regarded, like China in recent years, as a threat to American prosperity. “President Reagan’s pragmatism contrasted strongly with the utopian dreams of free traders,” Mr. Lighthizer wrote in a 2008 piece criticizing Senator John McCain, Republican of Arizona, for embracing “unbridled” free trade. Conservatives, he argued, “always understood that trade policy was merely a tool for building a strong and independent country with a prosperous middle class. """
def eval_lda(filename=sample_filename, n_topics=10):

	if len(sys.argv) == 2:
		filename = sys.argv[1]

	if len(sys.argv) == 3:
		try:
			filename = sys.argv[1]
			n_topics = int(sys.argv[2])
			print(n_topics)
		except:
			print("invalid integer")
			exit(1)

	vec, lda_object = lda(filename=filename, n_topics=n_topics)
	print_top_words(lda_object, vec.get_feature_names(), 40)
	print(lda_object.transform(vec.transform([s])))

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " | ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def load_dataset(filename):
	try:
		return pandas.read_csv(filename)
	except FileNotFoundError:
		print("File not found!")
		return None
	except Exception as e:
		print(str(e))
		return None

# assumes content is in column named # `content`
def lda(filename, n_topics=10):
	data = load_dataset(filename)
	data.dropna(subset=['content'], inplace=True)
	vec = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(2,2))
	transformed_docs = vec.fit_transform(data['content'])
	lda = LatentDirichletAllocation(n_components=n_topics, max_iter=100)
	lda.fit(transformed_docs)
	return vec, lda


if __name__ == "__main__":
	eval_lda(filename='breitbart_articles.csv')