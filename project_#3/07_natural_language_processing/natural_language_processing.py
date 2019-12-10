import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting a new result with Regression
predictions = classifier.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print(cm)