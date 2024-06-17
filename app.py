import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import os

st.set_page_config(layout="wide")

template_1 = """Give me a 100-word summary on the following blog post:

```
{text_1}
```
"""

template_2 = """Why the following summary is important:

```
{text_2}
```
"""

template_twitter = """Write a twitter post promoting the following blog:

```
{text_twitter}
```
"""

client = OpenAI()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text, section="Summarizer"):
    if st.session_state["section"] == "Summarizer":
        response_1 = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": template_1.format(text_1=input_text)}]
        )

        response_2 = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": template_2.format(text_2=response_1.choices[0].message.content)}]
        )

        res = [
            response_1.choices[0].message.content,
            response_2.choices[0].message.content
        ]
    elif st.session_state["section"] == "Twitter":
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": template_twitter.format(text_twitter=input_text)}]
        )

        res = [response.choices[0].message.content]

    return res

def get_openai_response(input_text):
    if st.session_state["section"] == "Summarizer":
        response_1 = genai.GenerativeModel("models/gemini-1.5-pro-001").generate_content(template_1.format(text_1=input_text))
        response_2 = genai.GenerativeModel("models/gemini-1.5-pro-001").generate_content(template_2.format(text_2=response_1.text))

        res = [
            response_1.text,
            response_2.text
        ]
    elif st.session_state["section"] == "Twitter":
        response = genai.GenerativeModel("models/gemini-1.5-pro-001").generate_content(template_twitter.format(text_twitter=input_text))
        res = [response.text]

    return res

def generate_completion(input_text):
    if input_text:
        if selected_model == "OpenAI API":
            return get_openai_response(input_text)
        elif selected_model == "Google Gemini API":
            return get_gemini_response(input_text)

    return []

st.title("Text Generation with OpenAI")

# LLM Selection
selected_model = st.selectbox("Select LLM", ["OpenAI API", "Google Gemini API"], key="selected_model")

section_index = 0
if "section" in st.session_state:
    section_index = ["Summarizer", "Twitter"].index(st.session_state["section"])

st.session_state["section"] = st.radio("Pick a section", ["Summarizer", "Twitter"], index=section_index)

# Create columns for input and output
left_col, right_col = st.columns(2)

if st.session_state["section"] == "Summarizer":
    with left_col:
        # Input text areas with state preservation
        input_texts = []
        for i in range(4):
            if f"input_text_{i}" not in st.session_state:
                st.session_state[f"input_text_{i}"] = ""  # Initialize state
            input_texts.append(
                st.text_area(f"Input Text {i+1}", key=f"input_text_{i}", height=200)
            )

        # Submit button
        if st.button("Generate Text"):
            st.session_state["completions"] = []
            for text in input_texts:
                if text:
                    st.session_state["completions"].append(generate_completion(text))

    with right_col:
        # Display generated text
        if "completions" in st.session_state:  # Check if completions are generated
            for i, completion in enumerate(st.session_state["completions"]):
                if completion is not None:
                    st.markdown(f"**Generated First Details {i+1}:**")
                    st.write(completion[0])
                    st.markdown(f"**Generated Second Details {i+1}:**")
                    st.write(completion[1])
                    st.divider()

elif st.session_state["section"] == "Twitter":
    twitter_completion = []
    with left_col:
        # TextArea Input Text
        if "twitter_input_text" not in st.session_state:
            st.session_state["twitter_input_text"] = ""  # Initialize state

        twitter_input_text = st.text_area(f"Twitter Input Text", key="twitter_input_text", height=200)

        # Submit Button
        if st.button("Generate Text"):
            if twitter_input_text:
                twitter_completion = generate_completion(twitter_input_text)

    with right_col:
        # Display generated text
        if len(twitter_completion) > 0:
            st.markdown("**Generated Details:**")
            st.write(twitter_completion[0])
