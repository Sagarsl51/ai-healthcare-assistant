import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_openai import ChatOpenAI
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "chatbot"
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name, embedding=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini")

st.set_page_config(page_title="AI Healthcare Assistant 🩺", page_icon="💊")

st.title("AI Healthcare Assistant 🩺💬")
st.write("Ask me any medical-related question. I’ll answer based on real medical docs.")

# Chat input
user_input = st.text_input("Your Question:", placeholder="e.g. What is acne?")

if st.button("Ask"):
    if user_input:
        results = docsearch.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        prompt = f"""
        You are a medical support assistant. Use the below medical context strictly to answer.

        Context:
        {context}

        Question:
        {user_input}

        Answer clearly, safely, and avoid self-diagnosis. 
        Recommend doctor visits when needed.
        """

        response = llm.invoke(prompt)
        st.subheader("🧠 Answer")
        st.write(response)

        st.subheader("📚 Sources")
        for doc in results:
            st.write(doc.metadata.get("source"))
    else:
        st.warning("Please enter a question.")
