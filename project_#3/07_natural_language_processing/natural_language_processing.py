import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

import datasets.dataset_provider as data_provider

dataset = data_provider.get_restaurant_reviews()

# Cleaning texts
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []
for rev in dataset['Review']:
    review = re.sub('[^a-zA-Z]', ' ', rev)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
