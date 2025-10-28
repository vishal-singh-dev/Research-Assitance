
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

from langchain_community.document_loaders import TextLoader
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chains import create_stuff_documents_chain
from langchain_community.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Milvus
from config.settings import HUGGINGFACEHUB_API_TOKEN


hf_token = HUGGINGFACEHUB_API_TOKEN

if not hf_token:
    raise ValueError("Missing Hugging Face API token. Please set HUGGINGFACEHUB_API_TOKEN in .env")

loader = TextLoader("./document.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
milvus_url = "tcp://localhost:19530"
collection_name = "text_chunks"

vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_args={"host": "localhost", "port": "19530"},
    collection_name=collection_name
)
model_name = "google/flan-t5-base"
pipe = pipeline("text2text-generation", model=model_name, tokenizer=model_name)
llm = HuggingFacePipeline(pipeline=pipe)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template("""
Answer the question using the context below.

Context:
{context}

Question:
{input}
""")

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)