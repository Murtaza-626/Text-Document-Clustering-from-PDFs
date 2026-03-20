import pdfplumber
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def extract_text_from_pdf(file):
    extracted_text = ""
    num_pages = 0
    
    try:
        with pdfplumber.open(file) as pdf:
            num_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
        return extracted_text.strip(), num_pages
    except Exception as e:
        return None, 0
    

def clean_and_lemmatize(text):
    # Cleaning raw text by removing noise and applying lemmatization.
    if not text:
        return ""
        
    # Converting text to lowercase
    text = text.lower()
    
    # Removing punctuation, special characters, and numbers (keeping only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenizing the text 
    tokens = nltk.word_tokenize(text)
    
    # Removing stopwords and apply Lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2 # Also dropping very short artifacts (1-2 letters)
    ]
    
    # Joining the tokens back into a single clean string
    return " ".join(cleaned_tokens)

