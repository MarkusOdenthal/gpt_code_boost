import streamlit as st
import tiktoken
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from gpt_code_boost.components.sidebar import sidebar
from gpt_code_boost.retrieval_answer.retrieval_answer import (
    format_query,
    system_message,
)

st.set_page_config(page_title="GPTCodeBoost", page_icon="👨‍💻", layout="wide")
st.header("👨‍💻GPTCodeBoost")

sidebar()


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method="markdown"):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


def get_text():
    input_text = st.text_area("You: ", "", key="input")
    return input_text


if "system" not in st.session_state:
    st.session_state["system"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "OPENAI_API_KEY" in st.session_state:
    open_api_key = st.session_state["OPENAI_API_KEY"]
else:
    open_api_key = None

# Layout of input/response containers
input_container = st.container()
response_container = st.container()
encoding_name = "cl100k_base"
encoding = tiktoken.get_encoding(encoding_name)

if open_api_key:
    query = get_text()
    ask_button = st.button("ask")

    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box, display_method="write")
    chat = ChatOpenAI(
        streaming=True,
        callbacks=[stream_handler],
        openai_api_key=open_api_key,
        verbose=True,
    )

    if query:
        if len(st.session_state["past"]) > 0:
            token_count = 0
            token_limit = 4000
            messages = []
            system_message = st.session_state["system"][0]
            messages.append(SystemMessage(content=system_message))
            token_count += len(encoding.encode(system_message))
            query_formatted = format_query(query, open_api_key)
            token_count += len(encoding.encode(query_formatted))
            for past_query, past_answer in zip(
                reversed(st.session_state["past"]),
                reversed(st.session_state["generated"]),
            ):
                token_count += len(encoding.encode(past_query))
                token_count += len(encoding.encode(past_answer))
                if token_count > token_limit:
                    # if we're over the token limit, stick with what we've got
                    break

                messages.append(HumanMessage(content=past_query))
                messages.append(AIMessage(content=past_answer))
            messages.append(HumanMessage(content=query_formatted))
        else:
            system_message = system_message(query, open_api_key)
            token_system_message = len(encoding.encode(system_message))
            st.session_state.system.append(system_message)
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=query),
            ]

        response = chat(messages)
        llm_response = response.content
        st.session_state.past.append(query)
        st.session_state.generated.append(llm_response)

    with st.expander("Conversation History", expanded=True):
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.info(st.session_state["past"][i], icon="🧐")
            st.success(st.session_state["generated"][i], icon="🤖")
else:
    st.warning(
        "API key required to try this app. The API key is not stored in any form."
    )
