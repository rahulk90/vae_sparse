from sklearn.feature_extraction.text import CountVectorizer

doclist_train = ['tractors are tools of agriculture', 'this is a sentence']
doclist_test  = ['NASA is a space agency', 'james bond is a book']

ctvec         = CountVectorizer(stop_words='english',analyzer='word',strip_accents='ascii')
ctvec.fit(doclist_train+doclist_test)
print ctvec.transform(doclist_train).toarray()
print ctvec.transform(doclist_test).toarray()
print ctvec.vocabulary_ 
