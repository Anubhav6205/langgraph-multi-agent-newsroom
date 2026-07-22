from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader
from logger import section, log_in, log_out


section("kb.py :: PDF load", "bold white")
pdf_loader = PyPDFLoader(os.path.join(os.path.dirname(__file__),"knowledge.pdf"))
raw_documents = pdf_loader.load()
log_out("raw_documents count", len(raw_documents))


section("kb.py :: text split", "bold white")
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50, separators=["\n\n","\n",",","."])
log_in("chunk_size", 500)
log_in("chunk_overlap", 50)
split_documents = text_splitter.split_documents(raw_documents)
log_out("split_documents count", len(split_documents))


section("kb.py :: embed → Chroma", "bold white")
persist_dir = os.path.join(os.path.dirname(__file__),"chroma_db")
log_in("persist_directory", persist_dir)
vectorstore=Chroma.from_documents(split_documents, embedding=OpenAIEmbeddings(), persist_directory=persist_dir)
log_out("vectorstore type", type(vectorstore).__name__)
