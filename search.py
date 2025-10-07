import json
import nltk

# Load tokenizer and stemmer (same as before)
tweet_tokenizer = nltk.TweetTokenizer()
porter_stemmer = nltk.PorterStemmer()

def normalize_and_stem(word):
    tokens = tweet_tokenizer.tokenize(word)
    tokens = [t.lower() for t in tokens if t.isalnum()]
    stems = [porter_stemmer.stem(t) for t in tokens]
    return stems

def load_index(index_path):
    """Load weighted index into dictionary: {word: [(doc_id, weight), ...]}"""
    index = {}
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            word = parts[0]
            postings = [(parts[i], float(parts[i+1])) for i in range(1, len(parts), 2)]
            index[word] = postings
    return index

def load_filenames(filenames_path):
    with open(filenames_path, "r", encoding="utf-8") as f:
        return json.load(f)

def search(query, index, filenames, top_k=10):
    stems = normalize_and_stem(query)
    scores = {}
    for stem in stems:
        if stem in index:
            for doc_id, weight in index[stem]:
                scores[doc_id] = scores.get(doc_id, 0) + weight
    
    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = [(filenames[f"doc{doc_id}"], score) for doc_id, score in ranked[:top_k]]
    return results

if __name__ == "__main__":
    index = load_index("./index_output/merged_weighted_index.txt")
    filenames = load_filenames("./index_output/filenames.json")
    
    while True:
        query = input("\nEnter your search query (or 'exit'): ")
        if query.lower() == "exit":
            break
        results = search(query, index, filenames)
        if results:
            print("\nTop results:")
            for title, score in results:
                print(f"- {title} (score: {score:.4f})")
        else:
            print("No results found.")
