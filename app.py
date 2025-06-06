import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import cohere
from langchain_chroma import Chroma 
from langchain_core.documents import Document


st.title("Semantic Search Engine")
st.header("upload a file to get started", divider="green")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 200, add_start_index = True
)
load_dotenv()
# Get API key from environment
api_key = os.getenv("COHERE_API_KEY")
# Initialize Cohere client
co = cohere.Client(api_key)
#Embedding Model 
# Initialize the embedding model
class CohereEmbedder:
    def __init__(self, client, model_name="embed-english-v3.0"):
        self.client = client
        self.model_name = model_name

    def embed_text(self, text):
        return self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_document"  # 👈 required for v3/v4 models
        ).embeddings[0]

    def embed_texts(self, texts):
        return self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document"
        ).embeddings
#for the model variable we create an instance  of the class above
embedding_model = CohereEmbedder(client=co)
#vector store
chroma_vector_store = Chroma(
    collection_name = "my_docs",
    embedding_function= embedding_model,
    persist_directory= "./chroma/db" 
)
uploaded_file = st.file_uploader("Select a file")

if uploaded_file is not None:
    with st.spinner("Processing file .."):
        try:
            print("File into", uploaded_file)

            #save file in memory
            temp_file_path = uploaded_file.name

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

                #PDF file loader
                loader = PyPDFLoader(temp_file_path)
                docs =loader.load()
                print("Docs :",docs)

                #create chunks
                chunks = text_splitter.split_documents(docs)
                print("Chunks create: ", len(chunks))
                
                for i, chunk in enumerate(chunks):
                    print(f"Chunk {i} is of size", len(chunk.page_content))

                #create embeddingd
                #embedder = CohereEmbedder(co)
                #for i, chunk in enumerate(chunks):
                   # embedding = embedder.embed_text(chunk.page_content)
                    #print(f"Chunk {i} embedding: {embedding[:5]}")
                chroma_ids = chroma_vector_store.add_documents(documents)


        except Exception as e:
            print(e) 