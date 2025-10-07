import streamlit as st
from search import search  # Import the search function
from search import load_index, load_filenames

# Load once
index = load_index("./index_output/merged_weighted_index.txt")
filenames = load_filenames("./index_output/filenames.json")

st.title("Newspaper Search Engine")

query = st.text_input("Enter your search query:")

if query:
    results = search(query, index, filenames)
    if results:
        st.subheader("Top Results")
        for title, score in results:
            st.write(f"- {title} (score: {score:.4f})")
    else:
        st.write("No results found.")
