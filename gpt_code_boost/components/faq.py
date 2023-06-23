import streamlit as st


def faq():
    st.markdown(
        """
# FAQ
## How does GPTCodeBoost work?
GPTCodeBoost works by first indexing the latest code documentation from GitHub. The documentation is broken down into 
smaller chunks and stored in a vector index that enables semantic search and retrieval. When you interact with 
ChatGPT through our tool, GPTCodeBoost will provide the context of the most recent updates to the library you're using
by querying this vector database. This process ensures that the coding recommendations provided by ChatGPT are accurate 
and up-to-date. 

## What do the numbers mean under each source?
For every code documentation, you'll see a citation number such as 3-12. The first number represents the section of the 
documentation, and the second number indicates the specific chunk within that section.

## Are the code recommendations 100% accurate?
While we strive to provide the most accurate code recommendations, they are not 100% accurate. GPTCodeBoost uses 
GPT-3.5 or GPT-4 to generate code suggestions, and while GPT-3.5 or GPT-4 is a powerful language model, they can 
sometimes make mistakes. 

In addition, GPTCodeBoost uses semantic search to find the most relevant chunks of code documentation but does not see 
the entire documentation, which means that it may not find all the relevant information for complex coding queries.

However, for most use cases, GPTCodeBoost is highly accurate and helpful in providing up-to-date code recommendations. 
Always cross-check with the latest library documentation for absolute certainty.
"""
    )
