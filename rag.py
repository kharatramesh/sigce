
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

# Running RAG 

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Set up Retriever and LLM
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model="gemma3:270m", temperature=0)

# 2. Define the Prompt Template
template = """You are a helpful assistant. Answer the question using ONLY the provided context below.
If you do not know the answer based on the context, say "I cannot find that in the documents."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 3. Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. Construct the RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Query the RAG system
query = "What is rollback?"
response = rag_chain.invoke(query)

print("--- Response ---")
print(response)
