# ============================================================
# AMAZON BOOK REVIEWS CLUSTERING PROJECT
# DATA PREPROCESSING MODULE
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import re
import string

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ============================================================
# LOAD STOPWORDS
# ============================================================

STOPWORDS = set(stopwords.words("english"))


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text: str) -> str:

    """
    Clean and preprocess review text.

    Steps:
    1. Lowercase conversion
    2. HTML removal
    3. URL removal
    4. Special character removal
    5. Punctuation removal
    6. Tokenization
    7. Stopword removal
    """

    try:

        # ====================================================
        # CONVERT TO STRING
        # ====================================================

        text = str(text)

        # ====================================================
        # LOWERCASE CONVERSION
        # ====================================================

        text = text.lower()

        # ====================================================
        # REMOVE HTML TAGS
        # ====================================================

        text = re.sub(
            r"<.*?>",
            "",
            text
        )

        # ====================================================
        # REMOVE URLS
        # ====================================================

        text = re.sub(
            r"http\S+",
            "",
            text
        )

        # ====================================================
        # REMOVE SPECIAL CHARACTERS & NUMBERS
        # ====================================================

        text = re.sub(
            r"[^a-zA-Z\s]",
            "",
            text
        )

        # ====================================================
        # REMOVE PUNCTUATION
        # ====================================================

        text = text.translate(
            str.maketrans(
                "",
                "",
                string.punctuation
            )
        )

        # ====================================================
        # TOKENIZATION
        # ====================================================

        tokens = word_tokenize(text)

        # ====================================================
        # REMOVE STOPWORDS
        # ====================================================

        tokens = [

            word for word in tokens

            if word not in STOPWORDS
        ]

        # ====================================================
        # REMOVE VERY SHORT WORDS
        # ====================================================

        tokens = [

            word for word in tokens

            if len(word) > 2
        ]

        # ====================================================
        # JOIN TOKENS
        # ====================================================

        cleaned_text = " ".join(tokens)

        return cleaned_text

    except Exception as e:

        print(f"Error in clean_text: {e}")

        return ""