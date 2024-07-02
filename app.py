import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import os, hmac


####################
# Authentication
####################

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["PASSWORD"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


################################
# Main Streamlit app starts here
################################

st.set_page_config(layout="wide")

template_1 = """{template_header}

```
{text_1}
```
"""

template_2 = """{template_header}

```
{text_2}
```
"""

template_twitter = """{template_header}

```
{text_twitter}
```
"""

client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])
genai.configure(api_key = st.secrets["GOOGLE_API_KEY"])

def get_gemini_response(input_text):
    # Call Google Gemini API
    res = []

    if st.session_state["section"] == "Summarizer":
        response_1 = client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {
                    "role": "user",
                    "content": template_1.format(text_1=input_text, template_header=st.secrets["TEMPLATE_1_HEADER"])
                }
            ]
        )

        response_2 = client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {
                    "role": "user",
                    "content": template_2.format(text_2=response_1.choices[0].message.content, template_header=st.secrets["TEMPLATE_2_HEADER"])
                }
            ]
        )

        res = [
            response_1.choices[0].message.content,
            response_2.choices[0].message.content
        ]
    elif st.session_state["section"] == "Twitter":
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {
                    "role": "user",
                    "content": template_twitter.format(text_twitter=input_text, template_header=st.secrets["TEMPLATE_TWITTER_HEADER"])
                }
            ]
        )

        res = [response.choices[0].message.content]

    return res

def get_openai_response(input_text):
    # Call OpenAI API
    res = []

    if st.session_state["section"] == "Summarizer":
        response_1 = genai.GenerativeModel(
                model_name = "models/gemini-1.5-pro-latest",
                generation_config = {"temperature": 0.1}
            ).generate_content(template_1.format(text_1=input_text, template_header=st.secrets["TEMPLATE_1_HEADER"]))

        response_2 = genai.GenerativeModel(
                model_name = "models/gemini-1.5-pro-latest",
                generation_config={"temperature": 0.1}
            ).generate_content(template_2.format(text_2=response_1.text, template_header=st.secrets["TEMPLATE_2_HEADER"]))

        res = [
            response_1.text,
            response_2.text
        ]
    elif st.session_state["section"] == "Twitter":
        response = genai.GenerativeModel(
                "models/gemini-1.5-pro-latest"
            ).generate_content(template_twitter.format(text_twitter=input_text, template_header=st.secrets["TEMPLATE_TWITTER_HEADER"]))

        res = [response.text]

    return res

def generate_completion(input_text):
    # LLM-Agnostic Completion Function
    if input_text:
        if selected_model == "OpenAI API":
            return get_openai_response(input_text)
        elif selected_model == "Google Gemini API":
            return get_gemini_response(input_text)

    return []

st.title("Text Generation with OpenAI")

# LLM Selection Dropdown
selected_model = st.selectbox(
        "Select LLM",
        ["OpenAI API", "Google Gemini API"],
        key = "selected_model"
    )

# Section Radio Button To Pick Section: "Summarizer" or "Twitter"
section_index = 0
if "section" in st.session_state:
    section_index = ["Summarizer", "Twitter"].index(st.session_state["section"])

st.session_state["section"] = st.radio(
    "Pick a section", ["Summarizer", "Twitter"], index=section_index)

# Each Section Has 2 Columns. Left Column For Input and Right Column For Output
left_col, right_col = st.columns(2)

if st.session_state["section"] == "Summarizer":
    # Summarizer Section
    with left_col:
        input_texts = []
        for i in range(4):
            if f"input_text_{i}" not in st.session_state:
                st.session_state[f"input_text_{i}"] = ""
            input_texts.append(
                st.text_area(f"Input Text {i+1}", key=f"input_text_{i}", height=200)
            )

        if st.button("Generate Text"):
            st.session_state["completions"] = []
            for text in input_texts:
                if text:
                    st.session_state["completions"].append(generate_completion(text))

    with right_col:
        if "completions" in st.session_state:
            for i, completion in enumerate(st.session_state["completions"]):
                if completion is not None:
                    st.markdown(f"**Generated First Details {i+1}:**")
                    st.write(completion[0])
                    st.markdown(f"**Generated Second Details {i+1}:**")
                    st.write(completion[1])
                    st.divider()

elif st.session_state["section"] == "Twitter":
    # Twitter Section
    twitter_completion = []
    with left_col:
        if "twitter_input_text" not in st.session_state:
            st.session_state["twitter_input_text"] = ""

        twitter_input_text = st.text_area(f"Twitter Input Text", key="twitter_input_text", height=200)

        if st.button("Generate Text"):
            if twitter_input_text:
                twitter_completion = generate_completion(twitter_input_text)

    with right_col:
        if len(twitter_completion) > 0:
            st.markdown("**Generated Details:**")
            st.write(twitter_completion[0])
