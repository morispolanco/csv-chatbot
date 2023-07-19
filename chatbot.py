from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import streamlit as st
from streamlit_chat import message
import tempfile

# Get OpenAI API key from user input
openai_api_key = st.sidebar.text_input(
    label="*****OpenAI API key*****",
    placeholder="ENTER OpenAI API key, sk-",
    type="password"
)

# Allow the user to upload a CSV file using Streamlit file uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    # Use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load the CSV file using CSVLoader and store the data
    csv_loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    csv_data = csv_loader.load()

    # Initialize embeddings, vectorstore, and chain for conversational retrieval
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(csv_data, embeddings)
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever()
    )

    def chat(query):
        # Perform conversational chat using the retrieval chain
        result = retrieval_chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Check and initialize session state variables if not present
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me about the " + uploaded_file.name + "!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hello"]

    # Create containers for chat history and user input
    response_container = st.container()
    input_container = st.container()

    with input_container:
        with st.form(key='my_form', clear_on_submit=True):
            # Retrieve user input via Streamlit text input
            user_input = st.text_input("Your query:", placeholder="Query your CSV data", key='input')
            submit_button = st.form_submit_button(label='Ask')

        if submit_button and user_input:
            # Call the chat function with user input and retrieve output
            output = chat(user_input)

            # Update session state with user input and generated output
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

            if st.session_state['generated']:
                # Display chat history in the response container
                with response_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
