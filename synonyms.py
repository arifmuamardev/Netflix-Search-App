import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

def get_synonyms(word):
    synonyms = set()
    synsets = wordnet.synsets(word)
    if synsets:
        for syn in synsets:
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return synonyms

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def expand_query(query):
    words = query.split()
    expanded_words = set(words.copy())
    
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    for word in words:
        # Stemming
        stemmed_word = ps.stem(word)
        expanded_words.add(stemmed_word)
        
        # Lemmatization
        pos = nltk.pos_tag([word])[0][1]
        wordnet_pos = get_wordnet_pos(pos)
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        expanded_words.add(lemmatized_word)
        
        # Synonyms
        synonyms = get_synonyms(word)
        expanded_words.update(synonyms)

    expanded_query = ' '.join(expanded_words)
    print(f"Original query: {query}")
    print(f"Expanded query: {expanded_query}")
    return expanded_query
