from dotenv import load_dotenv
import os

load_dotenv()
chave_openai = os.getenv("OPENAI_API_KEY")

import fitz 
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Especifique o caminho para o seu próprio PDF
pdf_path = r"C:\Users\Citel\Downloads\Notas da versão release 72.361.pdf"

# Extrair o texto do PDF
print("Extraindo o texto do PDF...")
doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text()

# Configurar o tokenizer para contar tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Função para contar tokens no texto
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Dividir o texto em partes menores usando RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Criar embeddings para as partes do texto usando OpenAIEmbeddings
print("Gerando embeddings e criando o banco de dados FAISS...")
embeddings = OpenAIEmbeddings(openai_api_key=chave_openai, model="text-embedding-ada-002")
db = FAISS.from_documents(chunks, embeddings)

# Função para realizar a busca de similaridade e responder perguntas
def responder_pergunta(pergunta):
    docs = db.similarity_search(pergunta)
    chain = load_qa_chain(OpenAI(openai_api_key=chave_openai, temperature=0), chain_type="stuff")
    resposta = chain.run(input_documents=docs, question=pergunta)
    return resposta

# Loop de perguntas interativas
while True:
    pergunta = input("Digite sua pergunta sobre o PDF (ou 'sair' para encerrar): ")
    if pergunta.lower() == "sair":
        print("Encerrando o programa.")
        break
    resposta = responder_pergunta(pergunta)
    print("Resposta:", resposta)
