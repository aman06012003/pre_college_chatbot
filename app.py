
import os
import pathlib
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GOOGLE_API_KEY"] = st.secrets["Gemini_key"]
# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_vectorstore_from_docx(docx_file_path):
    try:
        loader = Docx2txtLoader(docx_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
        document_chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            embedding=embeddings,
            documents=document_chunks,
            persist_directory="./data"
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "You are PreCollege AI assistant which helps students who have given jee mains exams which gives admission to premier institute of india.  you have real info provided above. for any other college give the info which is available if not available tell i am learning for that college please try again later. be interactive ask question from the student based on placement, college details and jee rank before proceeding to answer about college. Begin the conversation with a proper greeting and give the information of only that college which the user asks for. DO not give irrelevant answers. Try to be precise.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# RAG Chain
rag_prompt = PromptTemplate(
    template="""    
    You are PreCollege AI assistant, specialized in helping students who have given the JEE Mains exam to find the best colleges in India. 
    You have access to detailed information about various colleges, including placement data.
    Your task is to provide a concise and informative response to the query given by the user. 
    Always return the information in a clear and structured format.
    Try to stick to the question always and do not give irrelevant answers.
    If the user asks for specific colleges, provide detailed explanation with facts to answer that question. 
    If the user does not provide a JEE rank or branch preference, do not ask for it unless absolutely necessary. 
    Instead, provide general information about the colleges like its cut-off, placement records, fees, scholarship etc.
    Begin the conversation with a proper greeting.
    
    QUESTION: {question} \n
    CONTEXT: {context} \n
    Answer:
    """,
    input_variables=["question", "context"],
)

rag_prompt_chain = rag_prompt | llm | StrOutputParser()

def get_response(user_query, retriever_chain):
    formatted_chat_history = []
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            formatted_chat_history.append({"author": "user", "content": message.content})
        elif isinstance(message, SystemMessage):
            formatted_chat_history.append({"author": "assistant", "content": message.content})
    
    input_data = {
        "chat_history": formatted_chat_history,
        "input": user_query
    }
    
    CONTEXT = retriever_chain.invoke(input_data)
    result = rag_prompt_chain.invoke({"question": user_query, "context": CONTEXT})
    return result

st.set_page_config(page_title="College Data Chatbot")
st.title("PreCollege Chatbot")

# Automatically load and preprocess file from directory
docx_file_path = pathlib.Path("Chatbot Doc_updates.docx")  # Specify the file path

if not docx_file_path.exists():
    st.info("No .docx file found in the specified directory.")
else:
    st.session_state.docx_name = docx_file_path.name
    st.session_state.vector_store = get_vectorstore_from_docx(docx_file_path)
    if st.session_state.vector_store:
        st.success("Document processed successfully!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"author": "assistant", "content": "Hello, I am a bot. How can I help you?"}
    ]

if st.session_state.get("vector_store") is None:
    st.info("Document preprocessing failed or document not found.")
else:
    user_query = st.text_input("Type your message here...")
    if user_query:
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        response = get_response(user_query, retriever_chain)
        st.session_state.chat_history.append({"author": "user", "content": user_query})
        st.session_state.chat_history.append({"author": "assistant", "content": response})

    for message in st.session_state.chat_history:
        if message["author"] == "assistant":
            with st.chat_message("system"):
                st.write(message["content"])
        elif message["author"] == "user":
            with st.chat_message("human"):
                st.write(message["content"])
