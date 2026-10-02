
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load your local document
# loader = TextLoader("sample_knowledge.txt") # Or PyPDFLoader("manual.pdf")
pdf_files = [
	"Deployment.pdf", "devops.pdf", "docker.pdf"
]
documents = []
for pdf_file in pdf_files:
	documents.extend(PyPDFLoader(pdf_file).load())

# 2. Split document into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Total document chunks created: {len(chunks)}")

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Initialize local embedding model via Ollama
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# 2. Store embeddings locally on disk
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created and persisted locally.")
