import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

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
