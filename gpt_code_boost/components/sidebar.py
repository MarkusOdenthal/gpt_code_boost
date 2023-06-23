import streamlit as st

from gpt_code_boost.components.faq import faq


def set_openai_api_key(api_key: str):
    st.session_state["OPENAI_API_KEY"] = api_key


def set_model_typ(model_typ: str):
    st.session_state["MODEL_TYP"] = model_typ


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
            "2. Ask a question about the documentation💬\n"
            "3. Available Documentations: \n"
            "   - qdrant - v1.2.x\n"
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paste your OpenAI API key here (sk-...)",
            help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
            value=st.session_state.get("OPENAI_API_KEY", ""),
        )

        model_typ = st.selectbox("Please select a model.", ("gpt-3.5-turbo", "gpt-4"))

        if api_key_input:
            set_openai_api_key(api_key_input)

        if model_typ:
            set_model_typ(model_typ)

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "🚀GPTCodeBoost provides code recommendations powered by the latest library documentation, "
            "ensuring you're always up-to-date."
        )
        st.markdown("This tool is a work in progress. 🔨")
        st.markdown("Made by [MarkusOdenthal](https://twitter.com/MarkusOdenthal)")
        st.markdown("---")

        faq()
