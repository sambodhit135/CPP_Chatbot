    from langchain_huggingface import HuggingFaceEmbeddings
    import streamlit as st
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        try:
            from langchain.vectorstores import FAISS
        except Exception:
            try:
                from langchain.vectorstores.faiss import FAISS
            except Exception:
                raise

    st.set_page_config(page_title="C++ RAG Chatbot")
    st.title("C++ RAG Chatbot")
    st.write("Ask any question related to C++ Introduction")

    @st.cache_resource
    def load_vectorstore():
        loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        final_documents = text_splitter.split_documents(documents)

        # Embeddings (use full Hugging Face repo name)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create FAISS vector store
        db = FAISS.from_documents(final_documents, embeddings)
        return db

    db = load_vectorstore()

    query = st.text_input("Enter your question about C++:")
    if query:
        docs = db.similarity_search(query, k=3)  # fixed typo: similarity_search
        st.subheader("Retrieved Context:")
        for i, doc in enumerate(docs):
            st.markdown(f"Result {i+1}:")
            st.write(doc.page_content)
