import json
import os
import string
import nltk
import csv

# Increase CSV field size limit to handle large articles
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB

# Safely check and download punkt if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize tokenizer and stemmer
tweet_tokenizer = nltk.TweetTokenizer()
porter_stemmer = nltk.PorterStemmer()


def tokenize_line(line):
    """Tokenize a single line of text using TweetTokenizer."""
    return tweet_tokenizer.tokenize(line)


def normalize_tokens(tokens):
    """Normalize tokens: remove punctuation, numbers, lowercase first token."""
    for i, token in enumerate(tokens):
        if token in string.punctuation:
            continue
        if any(char.isnumeric() for char in token):
            continue
        if not any(char.isalnum() for char in token):
            continue
        yield token.lower() if i == 0 else token


def stem_tokens(tokens):
    """Apply Porter stemming to tokens."""
    for token in tokens:
        yield porter_stemmer.stem(token)


def create_filenames_json(output_dir, filenames):
    """Save mapping of 'doc{i}' -> filename or ID to JSON."""
    filenames_dict = {f"doc{i}": filename for i, filename in enumerate(filenames)}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "filenames.json"), "w", encoding="utf-8") as file:
        json.dump(filenames_dict, file, indent=4)


def process_csv_file(csv_path, filenames_json_output_dir, max_rows=None):
    """
    Process newspaper CSV, yield token-document ID pairs, and save filenames mapping.
    Expects CSV with columns: id, title, content
    """
    files_ids = []

    print("DEBUG: Started yielding tokens from CSV")

    with open(csv_path, "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)  # Use header names
        for i, row in enumerate(csv_reader):
            # Stop after max_rows if specified
            if max_rows is not None and i >= max_rows:
                break

            if i % 100 == 0:
                print(f"DEBUG: Yielding {i}th row")

            content = row.get("content", "")
            if content.strip():
                files_ids.append(row.get("title", f"doc_{i}"))

                tokens = tokenize_line(content)
                tokens = normalize_tokens(tokens)
                tokens = stem_tokens(tokens)

                for word in tokens:
                    if word:
                        yield [word, len(files_ids) - 1]

    print("DEBUG: Finished yielding tokens")
    create_filenames_json(filenames_json_output_dir, files_ids)
    print("DEBUG: Filenames JSON saved")
