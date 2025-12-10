"""
Utility functions for StackOverflow Tag Recommendation AI Model
Contains preprocessing, tokenization, and prediction helper functions
"""

import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Tuple
# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def text_preprocess(text: str) -> List[str]:
    """
    Preprocess text by cleaning, tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text: Raw text string to preprocess
        
    Returns:
        List of preprocessed tokens
    """
    global lemmatizer, stop_words
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return tokens


def text_to_numerical_sequence(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    """
    Convert tokenized text to numerical sequence.
    
    Args:
        tokens: List of text tokens
        vocab: Vocabulary dictionary
        max_len: Maximum sequence length
        
    Returns:
        List of token indices padded/truncated to max_len
    """
    # Convert tokens to indices
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Pad or truncate
    if len(sequence) < max_len:
        sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    
    return sequence



# Example usage and testing
if __name__ == "__main__":
    # Test preprocessing
    sample_text = "How do I read a CSV file in Python using pandas?"
    print("Original text:", sample_text)
    
    tokens = text_preprocess(sample_text)
    print("Preprocessed tokens:", tokens)
    
    # Test vocabulary building
    processed_texts = [
        ['how', 'read', 'csv', 'file', 'python'],
        ['python', 'pandas', 'dataframe'],
        ['csv', 'file', 'parsing']
    ]
    vocab = build_vocabulary(processed_texts, vocab_size=100)
    print(f"\nVocabulary size: {len(vocab)}")
    print("Sample vocab entries:", list(vocab.items())[:10])
    
    # Test sequence conversion
    sequence = text_to_numerical_sequence(tokens, vocab, max_len=20)
    print(f"\nNumerical sequence (length={len(sequence)}):", sequence)
