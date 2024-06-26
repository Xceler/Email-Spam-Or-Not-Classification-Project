import nltk
import re 
import pandas as pd 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , confusion_matrix

#Downloading NLTK
nltk.download('stopwords')
nltk.download('wordnet')


#Importing and Handling Csv file

data = pd.read_csv('email_classification.csv')
print(data.head())


corpus = []
wordnet = WordNetLemmatizer() 

for i in range(len(data)):

    #Removing Punctuation and others expect a-z , A-Z , Cleaning text data
    review = re.sub('[^a-zA-Z]', ' ', data['email'][i])

    #Lowercasing 
    review = review.lower()

    #Splitting 
    review = review.split() 

    #Lemmatization 
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]

    review = ' '.join(review)
    corpus.append(review)


#Vectorization 
tfidf = TfidfVectorizer()

#For Feature
x = tfidf.fit_transform(data['email'])

#For label 
y = pd.get_dummies(data['label'])
y = y.iloc[:, 1].values


#Splitting the data into train and test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Using Naive Bayes(MultiNomialNB)

nb = MultinomialNB()
model = nb.fit(x_train, y_train)

#Predict with x_test
y_pred = model.predict(x_test)

#Accuracy_score
accuracy = accuracy_score(y_pred, y_test)

#Confusion_Matrix
confusion = confusion_matrix(y_pred, y_test)

print(f"Accuracy Score: {accuracy}")
print(f"Confusion Matrix : {confusion}")
