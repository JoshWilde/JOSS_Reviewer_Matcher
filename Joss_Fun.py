# Imports
import numpy as np
import glob
import spacy

from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

df = np.load('Wikipedia_TF_IDF.npy',allow_pickle=True).item()

# Functions
def Get_Lemma_Words(POI_PDF):
    ''' 
    Parameters
    ----------
        POI_PDF : list
            A list containing a single string which is the contents of the paper
            
    Returns
    ----------
        words : array_like
            An array where each element is a processed word from the text
    ''' 
    text = str(POI_PDF)
    text2 = text.split() # splits the text into words
    words_no_punc = [] # defines an empty list

    for w in text2: # For each word in the text
        if w.isalpha(): # If the word is an alphanumberic value
            words_no_punc.append(w.lower()) # appends a lowercase version of the word to the no punctionation list
    from nltk.corpus import stopwords # Import stop words
    stopwords = stopwords.words('english')  # Defines english stop words
    clean_words = [] # define clean word list
    for w in words_no_punc: # for each word in no punctionation list
        if w not in stopwords: # if the word is not a stopword
            clean_words.append(w) # if the word is not a stopword it is appended to the clean word list
    clean_words_arr = '' # Defines an empty string
    for i in range(len(clean_words)): # For each word in clean words
        clean_words_arr = clean_words_arr + ' ' + str(clean_words[i]) # Appends the clean words to a string

    string_for_lemmatizing = clean_words_arr 
    lemmatizer = WordNetLemmatizer() 
    words_2 = word_tokenize(string_for_lemmatizing)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_2]

    lemmatized_words_arr = '' #  Defines an empty string
    for i in range(len(lemmatized_words)):  # For each word iin lemmanised words
        lemmatized_words_arr = lemmatized_words_arr + ' ' + str(lemmatized_words[i]) # Appends the lemmanised words to a string
    words = word_tokenize(lemmatized_words_arr) # Tokenises each word in the text
    return words


def Get_Top_Words_tf(Paper_interest, df=1, num_top20=20):
    ''' 
    Parameters
    ----------
        Paper_interest : string
            File path to the location of the PDF 
            
        df : 
            A
            
        num_top20 : Int
            Number of most frequent words that are used for calculating the vector of the paper
            
    Returns
    ----------
        top20_tf : array_like
            Array of the most frequent words from the paper in order
    ''' 
    POI_PDF = [extract_text(Paper_interest)] # Extracts text from the PDF file
    #text = str(POI_PDF)
    words =  Get_Lemma_Words(POI_PDF) # Lemmanises words from the extracted text
    top20_tf = -2 # If there are no lemmanised words, this function will output this value
    if len(words) > 0: # If there are lemmanised words
        fdist = FreqDist(words) # Calculates the frequency for each lemmanised word in the text
        X = np.array(fdist.most_common()) # Sorts the words in order of frequency
        top20_tf = X[:num_top20,0] # Saves the top N words as a list

    return top20_tf

def determine_wiki_td_idf(x, df=df):
    ''' 
    Parameters
    ----------
        x : string
            A

            
        df : dictionary
            A

            
    Returns
    ----------
        tf_idf_arr_names : array_like
            A
            
        tf_idf_arr_floats : array_like
            A
    ''' 
    tf_idf_arr_names = [] # Defining Empty array for the TF-IDF words
    tf_idf_arr_floats = [] # Defining Empty array for the TF-IDF scores
    for i in range(len(x)): # For each lemmanised word
        if x[i][0] in df: # If the lemmanised word is in the TF-IDF Dictionary
            wiki_tf = df[x[i][0]] # Find the IDF value for the word in the TF-IDF Dictionary
            doc_tf = int(x[i][1]) # Find the frequency of the word in this document
            tf_idf = np.log(wiki_tf/doc_tf) # Calculates an TF-IDF score
            tf_idf_arr_names.append(x[i][0])  # Appends the words to the array
            tf_idf_arr_floats.append(tf_idf) # Appends the TF-IDF score to the array
    return tf_idf_arr_names, tf_idf_arr_floats

def Get_Top_Words_tf_idf(Paper_interest, df, num_top20=20):
    ''' 
    Parameters
    ----------
        Paper_interest : string
            File path to the location of the PDF 
            
        num_top20 : Int
            Number of highest scoring words that are used for calculating the vector of the paper
            
    Returns
    ----------
        top20_tf_idf : array_like
            Array of the highest scoring words from the paper in order
    ''' 
    POI_PDF = [extract_text(Paper_interest)] # Extracts text from the PDF file
    #text = str(POI_PDF)
    words = Get_Lemma_Words(POI_PDF) # Lemmanises words from the extracted text
    fdist = FreqDist(words) # Calculates the frequency for each lemmanised word in the text
    X = np.array(fdist.most_common()) # Sorts the words in order of frequency
    tf_idf_arr_names, tf_idf_arr_floats = determine_wiki_td_idf(X, df=df) # Calculates the TF-IDF
    num_arr = np.array(tf_idf_arr_floats) # Converts the list to an array
    tf_idf_arr_names_arr = np.array(tf_idf_arr_names) # Converts the list to an array
    top20_tf_idf = tf_idf_arr_names_arr[np.argsort(num_arr)[:num_top20]] # Saves the top N words as an array

    
    return top20_tf_idf

def Generate_Paper_Vector(Paper_interest, model, df, get_word_fun=Get_Top_Words_tf, num_top20=20):
    ''' 
    Parameters
    ----------
        Paper_interest : string
            File path to the location of the PDF 
            
        model :
            A
        
        df : Dictionary
            A
            
        get_word_fun : function
            A
        
        num_top20 : int
            A
            
    Returns
    ----------
        pap_vector : array_like
            An array of shape (300) representing where the given paper lies in the
            model vector space.
            
        doc_top20 : string
            A string containing the 20 words that were 
        
    ''' 
    #average_vector = np.zeros((300)) # Creates an array for 300 zeros
    top20_tf = get_word_fun(Paper_interest, df, num_top20) # Gets the top N Words
    #print(top20_tf)
    doc_top20= '' # Creates empty string
    if top20_tf != -2: # If the paper has lemmanised words
        for i in top20_tf: # For each word in the top N
                doc_top20 = doc_top20 + i +' ' # Appends each top N word to list
    pap_vector = model(doc_top20).vector # generates a vector for the paper
    #average_vector = average_vector + pap_vector 
    
    return pap_vector, doc_top20

def Reviewer_Paper_Vector(paper_list, model, df, get_word_fun=Get_Top_Words_tf, num_top20=20):
    ''' 
    Parameters
    ----------
        Paper_interest : string
            File path to the location of the PDF 
            
        model : 
            A model that can generate a vector representation of the paper.
            
        df : Dictionary
            A 
            
        get_word_fun : 
            A function that will get the top N words, these could be defined as the 
            most frequent words, the highest score words in terms of TF-IDF, or any
            user defined function
            
        num_top20 : int
            A
                 
    Returns
    ----------
        average_vector : array_like
            A
            
    ''' 
    average_vector = np.zeros((300)) # Creates an array for 300 zeros
    mod = 0 # Keeps track of papers that do not add information to the average_vector
    for i in range(len(paper_list)): # For each paper in the list
        Paper_interest = paper_list[i] # Gets a paper path
        top20_tf = get_word_fun(Paper_interest, df, num_top20) # Generates the top N words for a paper
        doc_top20= ''  # Creates empty string
        if top20_tf != -2: # If the paper has lemmanised words
            for i in top20_tf: # For each word in the top N
                doc_top20 = doc_top20 + i +' ' # Append the top N words to a list
            pap_vector = model(doc_top20).vector # Generates the vector for a paper
            average_vector = average_vector + pap_vector # adds the result to the average
        else:
            mod = mod +1 # Adds a value indicating that the paper had no words, hence did not add
            # any information to the average_vector
            
    diver = len(paper_list)-mod # subtracks the modification from the paper list
    if diver ==0: # If no papers added to the average_vector
        diver = 1 # Updates the division to 1, to avoid an error
    average_vector = average_vector/diver # Average vector divided by papers that added information
    
    return average_vector

# Generate TF Vectors Author
def Author_vectors_TF(folder_names, model, num_top20=20, gen_ave_vec=Reviewer_Paper_Vector, directory_offset=21):
    ''' 
    Parameters
    ----------
        folder_names : array_like
            Array of folder paths, each folder should be the name of the author and should contain their papers in 
            PDF format. 
            
        gen_ave_vec : function
            A function to generate the vectors for paper that we are trying to find a reviewer for
        
        directory_offset : int
            A value that clips the file path to ensure that the keys for the author name will only contain the 
            author name.
                 
    Returns
    ----------
        Author_Dict : Dictionary
            A dictionary of vectors for each author. The keys are the names of the folders. The items are vectors
            of shape (300) which is the average vector for each authors work.
            
    ''' 
    Author_Dict = {} # Defines an empty dictionary
    for k in range(len(folder_names)): # For each author
        #print(folder_names[k][directory_offset:]+ ' - ' +str(k))
        paper_list = glob.glob(folder_names[k]+'/*.pdf') # Finds all PDF files in this folder
        print(paper_list)
        average_vector = gen_ave_vec(paper_list, model, num_top20) # Generates the average vector for all the papers in this folder
        Author_Dict[folder_names[k][directory_offset:]] = average_vector # Adds this average vector to the dictionary
    return Author_Dict

# Generate TF Vectors Paper
def Paper_vectors_TF(paper_list, model,num_top20=20, gen_pap_vec=Generate_Paper_Vector):
    ''' 
    Parameters
    ----------
        paper_list : array_like
            Array of file paths to PDF files
            
        gen_pap_vec : function
            A function to generate the vectors for paper that we are trying to find a reviewer for
            
    Returns
    ----------
        Paper_Dict : Dictionary
            All the keys should be the DOI numbers for each paper taken from the file name. The items are vectors
            of shape (300) which is the vector for where this paper lies in the model vector space.
            
        Paper_20_Dict : Dictionary
            All the keys should be the DOI numbers for each paper taken from the file name. The items are the 
            top 20 words from the paper that have been used to generate the vector representation.

    ''' 
    Paper_Dict = {}  # Defines an empty dictionary
    Paper_20_Dict = {}  # Defines an empty dictionary
    for k in range(len(paper_list)): # For each paper
        print(paper_list[k]+ ' - ' +str(k))
        paper_vector, doc_top20 = gen_pap_vec(paper_list[k], model, num_top20) # Generates paper vector and shows the top N words
        Paper_Dict[paper_list[k][-9:-4]] = paper_vector # Adds this vector to the dictionary
        Paper_20_Dict[paper_list[k][-9:-4]] = doc_top20 # Adds the top N words to the dictionary
    return Paper_Dict, Paper_20_Dict

# Generate TF-IDF Vectors Author
def Author_vectors_TF_IDF(folder_names, model, num_top20=20, gen_ave_vec=Reviewer_Paper_Vector, directory_offset=21):
    ''' 
    Parameters
    ----------
        folder_names : array_like
            Array of folder paths, each folder should be the name of the author and should contain their papers in 
            PDF format. 
            
        gen_ave_vec : function
            A function to generate the average vector for an author's work. This work should be in PDF format and
            the function will calculate the average vector for all works in this folder.
            
        directory_offset : int
            A value that clips the file path to ensure that the keys for the author name will only contain the 
            author name.
                 
    Returns
    ----------
        Author_Dict : Dictionary
            A dictionary of vectors for each author. The keys are the names of the folders. The items are vectors
            of shape (300) which is the average vector for each authors work.
            
    ''' 
    Author_Dict = {}  # Defines an empty dictionary
    for k in range(len(folder_names)): # For each author
    #k = k+260
        print(folder_names[k][directory_offset:]+ ' - ' +str(k))
        paper_list = glob.glob(folder_names[k]+'/*.pdf')  # Finds all PDF files in this folder
        average_vector = gen_ave_vec(paper_list, model,num_top20) # Generates the average vector for all the papers in this folder
        Author_Dict[folder_names[k][directory_offset:]] = average_vector # Adds this average vector to the dictionary
    return Author_Dict

# Generate TF-IDF Vectors Paper
def Paper_vectors_TF_IDF(paper_list, model, num_top20=20, gen_pap_vec=Generate_Paper_Vector):
    ''' 
    Parameters
    ----------
        paper_list : array_like
            Array of file paths to PDF files
        
        gen_pap_vec : function
            A function to generate the average vector for an author's work. This work should be in PDF format and
            the function will calculate the average vector for all works in this folder.
                 
    Returns
    ----------
        Paper_Dict : Dictionary
            All the keys should be the DOI numbers for each paper taken from the file name. The items are vectors
            of shape (300) which is the vector for where this paper lies in the model vector space.
            
        Paper_20_Dict : Dictionary
            All the keys should be the DOI numbers for each paper taken from the file name. The items are the 
            top 20 words from the paper that have been used to generate the vector representation.
            
    ''' 
    Paper_Dict = {}  # Defines an empty dictionary
    Paper_20_Dict = {}  # Defines an empty dictionary
    for k in range(len(paper_list)): # For each paper
        print(paper_list[k]+ ' - ' +str(k))
        paper_vector, doc_top20 = gen_pap_vec(paper_list[k],model,df, Get_Top_Words_tf_idf, num_top20) # Generates paper vector and shows the top N words
        Paper_Dict[paper_list[k][-9:-4]] = paper_vector # Adds this vector to the dictionary
        Paper_20_Dict[paper_list[k][-9:-4]] = doc_top20 # Adds the top N words to the dictionary
    return Paper_Dict, Paper_20_Dict

def Paper_cosine(author_keys, paper_vec, N=5, printer=True):
    ''' 
    Parameters
    ----------
        author_keys : Dictionary Keys
            A
             
        paper_vec : array-like
            A
            
        N : int
            Number of reviewers suggested 
            
        printer : Boolean
            A
            
    Returns
    ----------
        cos_sim_dict : dictionary
            A
    ''' 
    cos_sim_list = [] # Creates an empty list
    for i in range(len(author_keys)): # For each author key
        idx = list(author_keys)[i] # Creates an index
        author_vec = author_vectors[idx] # Loads the vector for the given author key
    
        cos_sim = cosine_similarity(np.array([paper_vec]), np.array([author_vec]))[0,0] # Calculates the cosine similarity 
        # of the paper and the author of the index
        cos_sim_list.append(cos_sim) # appends cosine similarity to a list of all cosine similarities for each author for 
        # this one paper
    cos_sim_list = np.array(cos_sim_list) # Converts list to numpy array

    cos_sim_dict = {} # Creates an empty dictionary
    sorted_idx = np.argsort(cos_sim_list)[-N:] # Sorts list and selects the top N highest scoring authors
    for i in range(N): # for each of the top N authors
        idx = sorted_idx[-i-1] # Creates an index
        doi = list(author_vectors)[idx] #Finds the author key for the high scoring cosine similarties
        if printer == True:
            print(doi + ' - ' + str(cos_sim_list[idx])[:6]) # Prints author key & cosine similarity for that author to the given paper
        cos_sim_dict[doi] = cos_sim_list[idx] # Adds the author key & cosine similarity to a dictionary
    return cos_sim_dict