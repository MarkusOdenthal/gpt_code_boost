import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from supabase import Client, create_client

from gpt_code_boost.chat.message_encoding import (
    get_encoding_length,
    query_message_prompt,
    system_message_prompt,
)
from gpt_code_boost.chat.stream_handler import StreamHandler
from gpt_code_boost.streamlit_components.sidebar import sidebar

supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

st.set_page_config(page_title="GPTCodeBoost", page_icon="👨‍💻", layout="wide")
st.header("👨‍💻GPTCodeBoost")

sidebar()


if "system" not in st.session_state:
    st.session_state["system"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "temp" not in st.session_state:
    st.session_state["temp"] = ""
if "OPENAI_API_KEY" in st.session_state:
    open_api_key = st.session_state["OPENAI_API_KEY"]
else:
    open_api_key = None


def clear_text():
    st.session_state["temp"] = st.session_state["query"]
    st.session_state["query"] = ""


# Layout of input/response containers
input_container = st.container()
response_container = st.container()


if open_api_key:
    st.text_area("You: ", "", key="query", on_change=clear_text)
    c1, c2, c3 = st.columns([0.5, 2, 3], gap="small")
    ask_button = c1.button("ask")
    c2.checkbox(
        "send details",
        True,
        key="send_details",
        help="allow question and the answer to be stored in the GPT-Code-Boost feedback database",
    )

    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box, display_method="write")
    chat = ChatOpenAI(
        streaming=True,
        callbacks=[stream_handler],
        openai_api_key=open_api_key,
        verbose=True,
    )

    if st.session_state.temp:
        if len(st.session_state["past"]) > 0:
            token_count = 0
            token_limit = 4000
            messages = []
            system_message = st.session_state["system"][0]
            messages.append(SystemMessage(content=system_message))
            token_count += get_encoding_length(system_message)
            query_formatted, docs = query_message_prompt(
                st.session_state.temp, open_api_key
            )
            token_count += get_encoding_length(query_formatted)
            for past_query, past_answer in zip(
                reversed(st.session_state["past"]),
                reversed(st.session_state["generated"]),
            ):
                token_count += get_encoding_length(past_query)
                token_count += get_encoding_length(past_answer)
                if token_count > token_limit:
                    # if we're over the token limit, stick with what we've got
                    break

                messages.append(HumanMessage(content=past_query))
                messages.append(AIMessage(content=past_answer))
            messages.append(HumanMessage(content=query_formatted))
        else:
            system_message, docs = system_message_prompt(
                st.session_state.temp, open_api_key
            )
            token_system_message = get_encoding_length(system_message)
            st.session_state.system.append(system_message)
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=st.session_state.temp),
            ]

        response = chat(messages)
        llm_response = response.content
        if st.session_state.send_details:
            insert_data = {
                "query": st.session_state.temp,
                "docs": docs,
                "llm_response": llm_response,
            }
            data, count = (
                supabase.table("streamlit_feedback").insert(insert_data).execute()
            )
        st.session_state.past.append(st.session_state.temp)
        st.session_state.generated.append(llm_response)

    with st.expander("Conversation History", expanded=True):
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.info(st.session_state["past"][i], icon="🧐")
            st.success(st.session_state["generated"][i], icon="🤖")
else:
    st.warning(
        "API key required to try this app. The API key is not stored in any form."
    )
