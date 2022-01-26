# JOSS_Reviewer_Matcher

## Premise of the project


## Ideas

### Idea 1 (Term Frequency)
The idea for this approach is to load in the pdf document as text and remove the stopwords from the text followed by lemmasation. The frequency of each word after this preprocessing is calculated. The N most frequent words are set aside, these words are considered to represent the overall topic of the paper.

The same preprocessing steps are followed on the domain topics of interest for the reviewers and editors. 

To pair these reviewers and editors to the paper, the most frequent N words are compared to the reviewer and editors preprocessed domain topic of interest. This is done by determining if any of the most frequent N words are present in the preprocessed domain topic of interest. For each match the editor/reviewer is given a +1 to their reviewer score.

The top X reviewer scores are presented to the user. 

The program will output the name of the suggested editors/reviewers, the amount of matched words they have, the matched words, and their domain topic of interest. This is to allow a human user to understand why the program has returned these editors/reviewers to confirm that the program has returned useful editors/reviewers.


### Idea 2 (Term Frequency - Inverse Document Frequency)
The idea for this approach is to load in the pdf document as text and remove the stopwords from the text followed by lemmasation. The frequency of each word after this preprocessing is calculated. 

A list of all words from English Wikipedia are loaded from a premade file. This contains the term frequency (TF) and the Term Frequency - Inverse Document Frequency (TF-IDF) for each word in English WIkipedia. 

The frequency of each word in the paper is compared to how often it appears in English WIkipedia. This is used to highlight the ‘rare’ words in this paper. These ‘rare’ words likely give a useful representation of the paper’s domain area. The top N ‘rarest’ words are set aside.

The same preprocessing steps are followed on the domain topics of interest for the reviewers and editors. 

To pair these reviewers and editors to the paper, the top N ‘rarest’ words are compared to the reviewer and editors preprocessed domain topic of interest. This is done by determining if any of the top N ‘rarest’ words are present in the preprocessed domain topic of interest. For each match the editor/reviewer is given a +1 to their reviewer score.

The top X reviewer scores are presented to the user. 

The program will output the name of the suggested editors/reviewers, the amount of matched words they have, the matched words, and their domain topic of interest. This is to allow a human user to understand why the program has returned these editors/reviewers to confirm that the program has returned useful editors/reviewers.


### Idea 3 (Word2Vec)
The idea for this approach is to load in the pdf document as text and remove the stopwords from the text followed by lemmasation. The frequency of each word after this preprocessing is calculated. The N most frequent words are set aside, these words are considered to represent the overall topic of the paper.

These N most frequent words are converted into vector space using Word2Vec on a pre-trained model (google-news-2013 or en-wikipedia). 

The same preprocessing steps are followed on the domain topics of interest for the reviewers and editors. This data are then converted into vector space using Word2Vec

To pair these reviewers and editors to the paper, the vector representations of the N most frequent words are compared to the vector representations of the editors/reviewers preprocessed domain topic of interest. 

The similarity of each vector from the Paper and each word from the preprocessed domain topic of interest is calculated (Using Cosine similarity). 

The top X editors/reviewers where the domain topic of interest is closest to the N most frequent words in the paper are presented to the user.  

The program will output the name of the suggested editors/reviewers,and why these editors were deemed the most suitable.


### Idea 4 (Average Word2Vec)
The idea for this approach is to load in the pdf document as text and remove the stopwords from the text (followed by lemmasation). 

All the words in this data are converted into vector space using Word2Vec on a pre-trained model (google-news-2013 or en-wikipedia). An average is taken of these vectors. This average vector should be ‘close’ to a word which ‘summaries’ the whole document. 
The same preprocessing steps are followed on the domain topics of interest for the reviewers and editors. This data are then converted into vector space using Word2Vec. (The domain topic vectors could also be averaged which should find a word which summaries the editors/reviewers).

To pair these reviewers and editors to the paper, the average paper vector is compared to the (average) editor/reviewer vector. 

The similarity of the average paper vector is compared to each vector from the editors/reviewers calculated (Using Cosine similarity). The more similar the average paper vector is to the (average) editor/reviewer vector the better the pairing.

The top X editors/reviewers are shown to the user

The program will output the name of the suggested editors/reviewers,and why these editors were deemed the most suitable.


### Idea 5 (Classification using Word2Vec and SCOPUS)
Using the approaches from either (Idea 4 or Idea 5) to determine vectors for the content from the paper. These vectors are compared to the Scopus Subject Areas and Science Journal Classification Codes. 

There are two main approaches I can see form this idea:

Compare the similarity of the paper vector to the vectors of Scopus Subject Areas and Science Journal Classification Codes. Then compare the editors/reviewers vectors to the vectors of Scopus Subject Areas and Science Journal Classification Codes. 

Turn this into a classification problem, give each paper a class corresponding to the Scopus Subject Areas and Science Journal Classification Codes. 

### Idea 6 (Doc2Vec)


### Idea 7 (GloVe)

### Idea 8 (BERT)

### Idea 9 (Sense2Vec)



