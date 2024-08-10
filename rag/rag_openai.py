import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load key
key_text = open("../../keys.txt", "r").read().strip()
for line in key_text.split("\n"):
    if line.startswith("OpenAI"):
        os.environ["OPENAI_API_KEY"] = line.split(": ")[-1].strip()

# Initialize OpenAI model
model = "gpt-4o-mini"
chat = ChatOpenAI(
    openai_api_key = os.environ["OPENAI_API_KEY"],
    model = model
)
messages = [
    SystemMessage(content="Welcome to the RAG chatbot! I'm here you help you extract vital information from documents."),
]

# Read the document and split by chunks
loader = PyPDFLoader("./materials/LLaMa3_paper.pdf")
pages = loader.load_and_split()     # Split by pages
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 5
)
docs = text_splitter.split_documents(pages)

# Embed
embed_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name="openai_embed")


def argument_prompt(query: str, k=3):
    # Find the top-l most relevant chunks based on similarity search
    results = vectorstore.search(query, search_type="similarity", k=k)
    source_knowledge = "\n".join([result.page_content for result in results])
    argument_prompt = f"""Use the contexts below to answer the query.
    
contexts:
{source_knowledge}

query:
{query}
"""
    return argument_prompt


query = "What are the numbers of parameters of models?"
prompt = HumanMessage(content=argument_prompt(query))
messages.append(prompt)
res = chat(messages)
print(res.content)