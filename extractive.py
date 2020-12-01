import numpy as np
import PyPDF2
import sys
import matplotlib.pyplot as plt
import networkx as nx
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer



def readDoc():
    name = input('Please input a file name: ') 
    print('You have asked for the document {}'.format(name))

    # now read the type of document
    if name.lower().endswith('.txt'):
        choice = 1
    elif name.lower().endswith('.pdf'):
        choice = 2
    else:
        choice = 3
        # print(name)
    print(choice)
    # Case 1: if it is a .txt file
        
    if choice == 1:
        f = open(name, 'r',encoding = "utf-8")
        document = f.read()
        f.close()
            
    # Case 2: if it is a .pdf file
    elif choice == 2:
        pdfFileObj = open(name, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pageObj = pdfReader.getPage(0)
        document = pageObj.extractText()
        pdfFileObj.close()
    
    # Case 3: none of the format
    else:
        print('Failed to load a valid file')
        print('Returning an empty string')
        document = ''
    
    print(type(document))
    return document




def tokenize(document):
    # We are tokenizing using the PunktSentenceTokenizer
    # we call an instance of this class as sentence_tokenizer
    doc_tokenizer = PunktSentenceTokenizer()
    
    # tokenize() method: takes our document as input and returns a list of all the sentences in the document
    
    # sentences is a list containing each sentence of the document as an element
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list




document = readDoc()
print('The length of the file is:', end=' ')
print(len(document))




sentences_list = tokenize(document)

# let us print the size of memory used by the list sentences
print('The size of the list in Bytes is: {}'.format(sys.getsizeof(sentences_list)))

# the size of one of the element of the list
print('The size of the item 0 in Bytes is: {}'.format(sys.getsizeof(sentences_list[0])))







input_words = []
input_sentences = len(sentences_list)
for sentence in sentences_list:
  for word in sentence.split(" "):
    input_words.append(word)

print(len(input_words))





print(type(sentences_list))






print('The size of the list "sentences" is: {}'.format(len(sentences_list)))







for i in sentences_list:
    print(i)






cv = CountVectorizer()
cv_matrix = cv.fit_transform(sentences_list)






cv_demo = CountVectorizer() # a demo object of class CountVectorizer

# I have repeated the words to make a non-ambiguous array of the document text matrix 

text_demo = ["Mohit is good, you are bad", "I am not bad"] 
res_demo = cv_demo.fit_transform(text_demo)
print('Result demo array is {}'.format(res_demo.toarray()))

# Result is 2-d matrix containing document text matrix
# Notice that in the second row, there is 2.
# also, bad is repeated twice in that sentence.
# so we can infer that 2 is corresponding to the word 'bad'
print('Feature list: {}'.format(cv_demo.get_feature_names()))







print('The data type of bow matrix {}'.format(type(cv_matrix)))
print('Shape of the matrix {}'.format(cv_matrix.get_shape))
print('Size of the matrix is: {}'.format(sys.getsizeof(cv_matrix)))
print(cv.get_feature_names())
print(cv_matrix.toarray())








normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())







print(normal_matrix.T.toarray)
res_graph = normal_matrix * normal_matrix.T
# plt.spy(res_graph)









nx_graph = nx.from_scipy_sparse_matrix(res_graph)
nx.draw_circular(nx_graph)
print('Number of edges {}'.format(nx_graph.number_of_edges()))
print('Number of vertices {}'.format(nx_graph.number_of_nodes()))
# plt.show()
print('The memory used by the graph in Bytes is: {}'.format(sys.getsizeof(nx_graph)))







ranks = nx.pagerank(nx_graph)

# analyse the data type of ranks
print(type(ranks))
print('The size used by the dictionary in Bytes is: {}'.format(sys.getsizeof(ranks)))

# print the dictionary
for i in ranks:
    print(i, ranks[i])






sentence_array = sorted(((ranks[i], s, i) for i, s in enumerate(sentences_list)), reverse=True)
sentence_array = np.asarray(sentence_array)







rank_max = float(sentence_array[0][0])
rank_min = float(sentence_array[len(sentence_array) - 1][0])







print(rank_max)
print(rank_min)








temp_array = []

# if all sentences have equal ranks, means they are all the same
# taking any sentence will give the summary, say the first sentence
flag = 0
if rank_max - rank_min == 0:
    temp_array.append(0)
    flag = 1

# If the sentence has different ranks
if flag != 1:
    for i in range(0, len(sentence_array)):
        temp_array.append((float(sentence_array[i][0]) - rank_min) / (rank_max - rank_min))

print(len(temp_array))








threshold = (sum(temp_array) / len(temp_array)) + 0.03





sentence_list = []
if len(temp_array) > 1:
    for i in range(0, len(temp_array)):
        if temp_array[i] > threshold:
                sentence_list.append((sentence_array[i][2], sentence_array[i][1]))
else:
    sentence_list.append(sentence_array[0][1])








# print(sentence_list)
# ERROR
sorted(sentence_list, key=lambda x: int(x[0]))
output_sentences = len(sentence_list)





output_words = []
for sentence in sentence_list:
  for word in sentence[1].split(" "):
    output_words.append(word)

len(output_words)





summary = " ".join(str(sentence) for index, sentence in sorted(sentence_list, key=lambda sentence_detail: int(sentence_detail[0])))
print(summary)
# save the data in another file, names sum.txt
f = open('outputofasb.txt', 'w')
#print(type(f))
f.write('\n')
f.write(summary)
f.write('\n\n')
f.write('Number of words in input file: ' + str(len(input_words)) + '\n')
f.write('Number of words in output file: ' + str(len(output_words)) + '\n')
f.write('Number of sentences in input file: ' + str(input_sentences) + '\n')
f.write('Number of sentences in output file: ' + str(output_sentences) + '\n')
f.close()
print(len(summary))

sentences_list = tokenize(summary)

# let us print the size of memory used by the list sentences
print('The size of the list in Bytes is: {}'.format(sys.getsizeof(sentences_list)))

print('The size of the list "sentences" is: {}'.format(len(sentences_list)))



