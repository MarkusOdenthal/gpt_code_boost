import qdrant_client
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant


def embedding_search(query, k, api_token):
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_token,
    )
    client = qdrant_client.QdrantClient(
        url=st.secrets["qdrant_url"], api_key=st.secrets["qdrant_api_key"]
    )
    db = Qdrant(client=client, embeddings=embeddings, collection_name="documentations")
    return db.similarity_search(query, k=k)


def format_context(docs):
    context = "\n\n".join(
        [
            f'From file {d.metadata["document_id"]}:\n' + str(d.page_content)
            for d in docs
        ]
    )
    print(context)
    return context


def system_message(query, api_token):
    docs = embedding_search(query, k=5, api_token=api_token)
    context = format_context(docs)

    prompt = """Given the following context and code, answer the following question. Do not use outside context, and do not assume the user can see the provided context. Try to be as detailed as possible and reference the components that you are looking at. Keep in mind that these are only code snippets, and more snippets may be added during the conversation.
    Reference the exact code snippets that you have been provided with. If you are going to write code, make sure to specify the language of the code. For example, if you were writing Python, you would write the following:

    ```python
    <python code goes here>
    ```

    Now, here is the relevant context: 

    Context: {context}
    """

    return prompt.format(context=context)


def format_query(query, api_token):
    docs = embedding_search(query, k=2, api_token=api_token)
    context = format_context(docs)
    return f"""Relevant context: {context}

    {query}"""
