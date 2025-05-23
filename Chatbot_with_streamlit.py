import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Load PDF, split, embed, and store in FAISS
def load_chunk_embed_vstore_file(path):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(path.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)

    return vectorstore

# Load free Hugging Face QA model
def load_llm():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Perform retrieval + QA
def rag_query(vectorstore, llm, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    result = llm(question=query, context=context)
    return result["answer"]

# Streamlit UI
st.title("📚 Technical Book RAG Chatbot (No OpenAI Key)")

uploaded_file = st.file_uploader("Upload a PDF book", type=["pdf"])

if uploaded_file and "vectorstore" not in st.session_state:
    with st.spinner("Processing PDF and creating knowledge base..."):
        st.session_state.vectorstore = load_chunk_embed_vstore_file(uploaded_file)
        st.session_state.llm = load_llm()
    st.success("Knowledge base created! You can now ask questions.")
if uploaded_file and "vectorstore" in st.session_state:
    query = st.text_input("Ask a question about the book:")

    if query:
        with st.spinner("Generating answer..."):
            answer = rag_query(st.session_state.vectorstore, st.session_state.llm, query)
        st.markdown("### Answer:")
        st.write(answer)
else:
    st.info("Please upload a PDF file to start.")