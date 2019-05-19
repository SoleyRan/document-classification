from gensim import corpora
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import collections
import csv
import pickle
csv_file = csv.reader(open('shuffled-full-set-hashed.csv','r'))
labels = []
contents = []
counts = collections.defaultdict(int)
print("Process begin...")
for l,c in csv_file:
    if len(c) == 0 or l not in ['DECLARATION', 'EXPIRATION NOTICE', 'BILL BINDER', 'APPLICATION', 'NON-RENEWAL NOTICE', 'INTENT TO CANCEL NOTICE', 'DELETION OF INTEREST', 'BINDER', 'POLICY CHANGE', 'RETURNED CHECK', 'REINSTATEMENT NOTICE', 'CHANGE ENDORSEMENT', 'BILL', 'CANCELLATION NOTICE']:
        continue
    counts[l] += 1
    labels.append(l)
    contents.append(c.split(' '))
print("In the data file given, we have:")
for l in counts:
    print("%s : %d" % (l, counts[l]))

# Create Dictionary
dictionary = corpora.Dictionary(contents)
# Treat data occurs less then 100 times and data with more than 80% frequency as no-use data or 'noise'
dictionary.filter_extremes(no_below=100, no_above=0.8)
dictionary.compactify()
dt = dictionary.token2id
length = len(dt)


# Sentence Vector
ls_of_wid = []
for words in contents:
    vector = [0] * length
    for word in words:
        if word in dt:
            vector[dt[word]] += 1
    ls_of_wid.append(vector)

# Divide data into train_cases and test_cases in 8:2
print("Training and Test data will be at a ratio of 8:2")
train_labels, test_labels, train_wids, test_wids = train_test_split(labels, ls_of_wid, test_size = 0.2)


# Train Naive Bayes
print("Training Naive Bayes...")
classifier = MultinomialNB()
classifier.fit(train_wids, train_labels)

# Predict the Model
print("Predicting the model...")
prediction = classifier.score(test_wids, test_labels)
print('Accuracy is:', prediction)

# Save the Model
with open('model.pickle', 'wb') as f:
    pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()
with open('dictionary.pickle', 'wb') as f:
    pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()

