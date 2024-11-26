from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#Preprocessing the given data
def preprocessing(sentence):
    new_sentence = sentence[:len(sentence) - 3]
    return new_sentence.lower()
# Load the data X-statement Y-label
X = []
Y = []
filenames = ['amazon data.txt', 'imdb data.txt','yelp data.txt']
for f in filenames:
    file = open(f, 'r') # r means read because you reading the data form the dataset
    lines = file.readlines()
    for line in lines:
        sentence = preprocessing(line)
        X.append(sentence)
        Y.append(line[-2])
    file.close()

    
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=11)
# Vectorize text data to numbers
vec = CountVectorizer(stop_words='english')
x_train = vec.fit_transform(x_train).toarray() 
x_test = vec.transform(x_test).toarray()
model = MultinomialNB()
model.fit(x_train, y_train)
# Results of the model
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
print("Accuracy:", score)
#Confusion matrix
matrix = confusion_matrix(y_test,y_pred)
print('Confusion matrix : \n',matrix)

#Prediction
comment=['this movie is bad ']
res=vec.transform(comment).toarray()
model.predict(res)
