import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load key
key_text = open("../../keys.txt", "r").read().strip()
for line in key_text.split("\n"):
    if line.startswith("Huggingface"):
        login(line.split(": ")[-1].strip())

# Read the document and split by chunks
loader = PyPDFLoader("./materials/LLaMa3_paper.pdf")
pages = loader.load_and_split()     # Split by pages
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 5
)
docs = text_splitter.split_documents(pages)

# Note that this sentence-transformers model is only used for embedding, which serves to find 
# the most relevant chunks. The actual answer generation is done by the latter model.
embedding_model_name = "sentence-transformers/sentence-t5-large"
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = Chroma.from_documents(documents=docs, embedding=embedding, collection_name="hf_embed")
query = "What are the numbers of parameters of models?"

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
prompt = argument_prompt(query)


# Answer the prompt
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
res_ids = model.generate(
    prompt_ids,
    max_new_tokens=100,
    temperature=0.7
)
res = tokenizer.decode(res_ids[0][prompt_ids.shape[-1]:], skip_special_tokens=True)
print(res)