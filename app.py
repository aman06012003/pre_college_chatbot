import streamlit as st
import pathlib
import textwrap
import google.generativeai as genai

genai.configure(api_key='AIzaSyCvkV4v4NPnPE2TcDGpIaJx56OIf_vUCnU')

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
            """You are an AI assistant with access to information about Indian Institutes of Information Technology (IIITs). Answer user queries based on the document. Only retrieve and provide specific details relevant to the user's query. For casual conversations, respond politely without mentioning specific IIITs unless asked.
            When a user asks for specific information about any IIIT or topic, provide only the details directly related to their query. Do not assume a specific IIIT unless the user specifies one.""",
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
