import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import os

st.set_page_config(layout="wide")

template_1 = """"Give me a 5-point summary on the following article:

```
{text_1}
```
"""

template_2 = """"Why the following points are important:

```
{text_2}
```
"""

client = OpenAI()  # For OpenAI API access
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # For Google Gemini API access

def get_gemini_response(input_text):
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

    return res

def get_openai_response(input_text):
    response_1 = genai.GenerativeModel("models/gemini-1.5-pro-001").generate_content(template_1.format(text_1=input_text))
    response_2 = genai.GenerativeModel("models/gemini-1.5-pro-001").generate_content(template_2.format(text_2=response_1.text))

    res = [
        response_1.text,
        response_2.text
    ]

    return res

def generate_completion(input_text):
    if input_text:
        if selected_model == "OpenAI API":
            res = get_openai_response(input_text)
            return res
        elif selected_model == "Google Gemini API":
            res = get_gemini_response(input_text)
            return res


st.title("Text Generation with OpenAI")

# LLM Selection
selected_model = st.selectbox("Select LLM", ["OpenAI API", "Google Gemini API"], key="selected_model")


# Create columns for input and output
left_col, right_col = st.columns(2)

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
                st.markdown("---")
                st.markdown(f"**Generated Second Details {i+1}:**")
                st.write(completion[1])
                st.divider()
