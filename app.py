import streamlit as st
import pathlib
import textwrap
import google.generativeai as genai

genai.configure(api_key=st.secrets["Gemini_key"])

model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

import docx2txt

file_path = pathlib.Path('Chatbot Doc.docx')
content = docx2txt.process(file_path)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("system"):
    st.write("Hi there!👋I'm ready to answer your questions. Ask me anything!")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your query"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response1 = model.generate_content([
            """You are PreCollege AI assistant which helps students who have given jee mains exams which gives admission to premier institute of india.  you have real info provided above. for any other college give the info which is available if not available tell i am learning for that college please try again later. be interactive ask question from the student based on placement, college details and jee rank before proceding to answer about college.""",
            f"Question: {prompt}",
            f"Context: {content}"
        ])
    response = f"{response1.text}"
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# with st.chat_message("assistant"):
#         st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})
