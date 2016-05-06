import pandas as pd      # Pandas open source library needed 
train = pd.read_csv("training data.csv", header=0)
                    
# Import BeautifulSoup 
from bs4 import BeautifulSoup # Beautiful Soup library needed            

import re

import nltk # Natural Language Toolkit library needed

from nltk.corpus import stopwords # Get the list of stopwords

# def load_file():
    # with open('review1.csv') as csv_file:
        # reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        # reader.next()
        # data =[]
        # target = []
        # for row in reader:
            # # skip missing data
            # if row[0] and row[1]:
                # data.append(row[0])
                # target.append(row[1])

        # return data,target
def preprocess( original_data ):
    # Using this function, we extract words from the raw data
    # Convert each input review into one string 
    # the output is a single string (a preprocessed movie review)
    #
    # Stripping all the HTML tags
    remove_html = BeautifulSoup(original_data,"lxml").get_text() 
    #
    # Stripping all the punctuations       
    remove_non_letters = re.sub("[^a-zA-Z]", " ", remove_html) 
    #
    # Change all the letters into lower case
    terms = remove_non_letters.lower().split()                             
    #
    # Store all the stop words in a set
    set_stopwords = set(stopwords.words("english"))                  
    # 
    # Remove all the stop words
    remove_stopwords = [w for w in terms if not w in set_stopwords]   
    #
    # The output is one string for the corresponding input review string
    return( " ".join( remove_stopwords )) 
    

new_data = preprocess( train["review"][0] )
print new_data  

# Compute the total number of reviews
total_reviews = train["review"].size

# Store all the processed reviews
print "Cleaning and parsing the training set movie reviews...\n"
processed_data = []

# Do this for all the reviews using a for loop
for i in xrange( 0, total_reviews ):
   
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, total_reviews )
    processed_data.append( preprocess( train["review"][i] ) )

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer # scikit-learn library needed

# Using scikit-learn's
# bag of words tool.  
init_vect = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 3500) 


#pass a list of strings as input to this function
#this function extracts the features
create_feat_vectors = init_vect.fit_transform(processed_data)

# represent all the features in the form of
# an array
create_feat_vectors = create_feat_vectors.toarray()

# Print the final words created in the vocabulary
final_words = init_vect.get_feature_names()
print final_words

import numpy as np

# Compute the frequency of each of these words
# in the vocabulary
frequency_words = np.sum(create_feat_vectors, axis=0)

# After computing the frequency of each of these words
# in the vocabulary
# print the count
for tag, count in zip(final_words, frequency_words):
    print count, tag

#Implementing Naive Bayes
 
#Use Gaussian probability density function to calculate probability
#Find the class probabilities
def Compute_Prob(original_data, inputVector):
	probabilities = {}
	 for classValue, classoriginal_data in original_data.iteritems():
		probabilities[classValue] = 1
		 for i in range(len(classoriginal_data)):
			 mean, stdev = classoriginal_data[i]
			 x = inputVector[i]
			 probabilities[classValue] *= calculateProbability(x, mean, stdev)
	 return probabilities

# Find out largest probability
# Using this make a prediction
def make_prediction(original_data, inputVector):
	 probabilities = calculateClassProbabilities(original_data, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			 bestProb = probability
			 bestLabel = classValue
	 return bestLabel
	
 def Classifier2(processed_data,target):
     # Performing cross validation 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.4,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
     evaluate_model(target_test,predicted)

    
print "Implementing the Random Forest classifier"
from sklearn.ensemble import RandomForestClassifier

# Hundred decision trees for the classifier
classifier = RandomForestClassifier(n_estimators = 100) 


# Pass the features to the classifier
# along with sentiment tags
classifier = classifier.fit( create_feat_vectors, train["sentiment"] )

# Read the test data
print "Reading the given test data"
test = pd.read_csv("test data final confirmed.csv", header=0 )

# Check the dimensions
# 25,000 Rows
# 2 Columns
print test.shape

# Processing reviews from the test data
total_reviews = len(test["review"])
processed_reviews = [] 

# Loop through the reviews in the test data
print "Processing the movie reviews present in the test data set...\n"
for i in xrange(0,total_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n done" % (i+1, total_reviews)
    new_data = preprocess( test["review"][i] )
    processed_reviews.append( new_data )

# Repeat the bag of words process just like ealier
bag_of_words = init_vect.transform(processed_reviews)
# Represnt in the form of array
bag_of_words = bag_of_words.toarray()

# Predict the sentiment of a test review 
prediction = classifier.predict(bag_of_words)

# Use pandas dataframe 
# Output should be contain
# One column with and another with sentiment (binary)
output = pd.DataFrame( data={"id":test["id"], "sentiment":prediction} )

# convert to csv format
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
