# Import necessary modules
import pandas as pd
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter

home_privacy = "We value and respect your privacy. To safeguard your personal details, we utilize the hashed value of your OpenAI API Key, ensuring utmost confidentiality and anonymity. Your API key facilitates AI-driven features during your session and is never retained post-visit. You can confidently fine-tune your research, assured that your information remains protected and private."

import contextlib
import logging
import subprocess
import sys
import time

import tqdm

# Unix, Windows and old Macintosh end-of-line
NEWLINES = ["\n", "\r\n", "\r"]


def unbuffered(proc, stream="stdout"):
    stream = getattr(proc, stream)
    with contextlib.closing(stream):
        while True:
            out = []
            last = stream.read(1)
            # Don't loop forever
            if last == "" and proc.poll() is not None:
                break
            while last not in NEWLINES:
                # Don't loop forever
                if last == "" and proc.poll() is not None:
                    break
                out.append(last)
                last = stream.read(1)
            out = "".join(out)
            yield out


def example():
    pass


def mod_install(args):
    """check if pip has submodules for progress bar iteration, get total of submodules to install that are not already
    instaled ( the count that will be installed ) for range, then update bar.. this would have to be done on each module
    """
    with subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    ) as p:
        for line in unbuffered(p):
            print(line)


def install_modules(modules, reinstall=False, update=False):
    """
    Install modules in the current interpreter.
    :param modules: An iterable of modules to install, e.g. ['torch', 'fastai']
    :param reinstall: Whether to reinstall already installed modules.
    :param update: Whether to update already installed modules.
    """
    progress_bar = tqdm.tqdm(total=len(modules), leave=False, desc="Installing pip modules")

    with progress_bar:
        for module in modules:
            """create unique bar for each module in modules list"""
            try:
                if not reinstall and not update and module in sys.modules:
                    print(f"{module} is already installed")
                    continue
                args = [sys.executable, "-m", "pip", "install"]
                if reinstall:
                    args.append("--force-reinstall")
                if update:
                    args.append("--upgrade")
                args.append(module)
                mod_install(args)
                time.sleep(1)
            except subprocess.CalledProcessError:
                logging.exception("Error installing {}", module)
            except KeyboardInterrupt:
                logging.exception("Interrupted, removing {}", module)
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", module])
            else:
                # optional additional callback function
                # this will install dependencies
                sys.stdout.write("\r")
                sys.stdout.flush()


install_modules(["markdown"])

HEADER = st.empty()




keycheck = st.sidebar.text_input("Enter Your OpenAI API Key:", type="password")
if st.secrets.password == keycheck:
    OPENAI_API_KEY = st.secrets.key
else:
    OPENAI_API_KEY = keycheck

st.sidebar.subheader("Setup")
st.sidebar.markdown("Get your OpenAI API key [here](https://platform.openai.com/account/api-keys)")
st.sidebar.divider()
st.sidebar.subheader("Model Selection")
llm_model_options = ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4']  # Add more models if available
model_select = st.sidebar.selectbox('Select LLM Model:', llm_model_options, index=0)
st.sidebar.markdown("""\n""")
temperature_input = st.sidebar.slider('Set AI Randomness / Determinism:', min_value=0.0, max_value=1.0, value=0.5)
st.sidebar.markdown("""\n""")
clear_history = st.sidebar.button("Clear conversation history")


if "conversation" not in st.session_state:
    st.session_state.conversation = None



# Splits a given text into smaller chunks based on specified conditions
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_markdown_text(markdown_docs):
    # Load Markdown documents
    DOCUMENTS = []
    TEXT_SPLITTER = MarkdownTextSplitter(chunk_size=355, chunk_overlap=20)
    localprog = HEADER.progress(0.0, "Loading Markdown Documents")
    size = len(markdown_docs)
    i = 0.0
    for file in markdown_docs:
        i += 1/size
        localprog.progress(i)
        loader = UnstructuredMarkdownLoader(file)
        docs = loader.load_and_split(text_splitter=TEXT_SPLITTER)
        DOCUMENTS.extend(docs)
    localprog.progress(1.0)
    HEADER.empty()
    return DOCUMENTS


# Generates embeddings for given text chunks and creates a vector store using FAISS
def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initializes a conversation chain with a given vector store
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=temperature_input, model_name=model_select),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h : h,
        memory=memory
    )
    return conversation_chain


# Upload file to Streamlit app for querying
user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
if user_uploads is not None:
    if st.button("Upload"):
        with st.spinner("Processing"):
            text_chunks =  get_markdown_text(user_uploads)

            # Create FAISS Vector Store of PDF Docs
            vectorstore = get_vectorstore(text_chunks)

            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)



# Initialize chat history in session state for Document Analysis (doc) if not present
if 'doc_messages' not in st.session_state or clear_history:
    # Start with first message from assistant
    st.session_state['doc_messages'] = [{"role": "assistant", "content": "Query your documents"}]
    st.session_state['chat_history'] = []  # Initialize chat_history as an empty list

# Display previous chat messages
for message in st.session_state['doc_messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# If user provides input, process it
if user_query := st.chat_input("Enter your query here"):
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    # Add user's message to chat history
    st.session_state['doc_messages'].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Generating response..."):
        # Check if the conversation chain is initialized
        if 'conversation' in st.session_state:
            st.session_state['chat_history'] = st.session_state.get('chat_history', []) + [
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            # Process the user's message using the conversation chain
            result = st.session_state.conversation({
                "question": user_query,
                "chat_history": st.session_state['chat_history']})
            response = result["answer"]
            # Append the user's question and AI's answer to chat_history
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": response
            })
        else:
            response = "Please upload a document first to initialize the conversation chain."

        # Display AI's response in chat format
        with st.chat_message("assistant"):
            st.write(response)
        # Add AI's response to doc_messages for displaying in UI
        st.session_state['doc_messages'].append({"role": "assistant", "content": response})
