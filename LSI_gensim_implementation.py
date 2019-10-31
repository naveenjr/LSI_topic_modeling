import nltk
import re
import numpy as np
from gensim import corpora, similarities
from gensim.models import LsiModel
from nltk.corpus import stopwords
#nltk.download("stopwords")
#nltk.download("wordnet")


# # steming and stopwords

# In[72]:


def filter_words_and_get_word_stems(document, word_tokenizer, word_stemmer,stopword_set, pattern_to_match_words=r"[^\w]",word_length_minimum_n_chars=2):

    document = re.sub(pattern_to_match_words, r" ", document)
    document = re.sub(r"\s+", r" ", document)
    words = word_tokenizer.tokenize(document)
    words_filtered = [word.lower()
                      for word in words
                      if word.lower() not in stopword_set and len(word) >= word_length_minimum_n_chars]
    word_stems = [word_stemmer.lemmatize(word) for word in words_filtered]
    return(word_stems)


# # input document and test doc




import os
inputDir = 'C:\\Users\\r-naveenj\\Desktop\\information_retrive\\LSI_for_docs\\texts\\'
text_files = os.listdir(inputDir)
documents_train = []
for f in text_files:
    file = open(os.path.join(inputDir, f), 'r', encoding = 'utf-8',errors='ignore')
    documents_train.append(file.read())

#document_test = "Data scientists program computers in various computer languages."
document_test = "FOOT ML80 IPR GROUND TOP MEA UNIT PICK UP HOU PMO M IDLE PICK UP RUBBER PICK UP HOUSING"


# # preprocess

# set stopword set, word stemmer and word tokenizer
stopword_set = set(stopwords.words("english"))
word_tokenizer = nltk.tokenize.WordPunctTokenizer()
word_stemmer =  nltk.WordNetLemmatizer()
#apply cleaning, filtering and word stemming to training documents
word_stem_arrays_train = [
        filter_words_and_get_word_stems(
                str(document),
                word_tokenizer,
                word_stemmer,
                stopword_set
                ) for document in documents_train]

#print("Word Stems of Training Documents:", word_stem_arrays_train)


word_stem_array_test = filter_words_and_get_word_stems(
        document_test,
        word_tokenizer,
        word_stemmer,
        stopword_set)
print("Word Stems of Test Document:", word_stem_array_test)


# # dictionary


dictionary = corpora.Dictionary(
  word_stem_array_train
  for word_stem_array_train in word_stem_arrays_train)
print("Dictionary :", dictionary)


corpus = [
  dictionary.doc2bow(word_stem_array_train) #check
  for word_stem_array_train in word_stem_arrays_train]
#print("Corpus :", corpus)



lsi_model = LsiModel(corpus=corpus,
        id2word=dictionary #, num_topics = 2 #(opt. setting for explicit dim. change)
        )
#print("Derivation of Term Matrix T of Training Document Word Stems: ",lsi_model.get_topics())

#print("Derivation of Term Matrix T of Training Document Word Stems: ",lsi_model.get_topics())
#Derivation of Term Document Matrix of Training Document Word Stems = M' x [Derivation of T]
#print("LSI Vectors of Training Document Word Stems: ",[lsi_model[document_word_stems] for document_word_stems in corpus])


#calculate cosine similarity matrix for all training document LSI vectors
cosine_similarity_matrix = similarities.MatrixSimilarity(lsi_model[corpus])
print("Cosine Similarities of LSI Vectors of Training Documents:",
      [row for row in cosine_similarity_matrix])


#calculate LSI vector from word stem counts of the test document and the LSI model content
vector_lsi_test = lsi_model[dictionary.doc2bow(word_stem_array_test)]
print("LSI Vector Test Document:", vector_lsi_test)

#perform a similarity query against the corpus
cosine_similarities_test = cosine_similarity_matrix[vector_lsi_test]
print("Cosine Similarities of Test Document LSI Vectors to Training Documents LSI Vectors:",
      cosine_similarities_test)

#OUTPUT
#get text of test documents most similar training document
#most_similar_document_test = documents_train[np.argmax(cosine_similarities_test)]
#most_similar_document_test = documents_train[np.argwhere(cosine_similarities_test >0.01]
most_similar_document_test = np.argwhere(cosine_similarities_test>.1)
#most_similar_document_test =  sorted(cosine_similarities_test, key=lambda x: x[1], reverse=True)
#print("Most similar Training Document to Test Document:", most_similar_document_test)
print(most_similar_document_test)

np.argwhere(cosine_similarities_test>.1)

most_similar_document_test = np.sort(cosine_similarities_test)[::-1]
print(most_similar_document_test)


np.argsort(cosine_similarities_test)[::-1]
