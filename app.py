import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether

load_dotenv()

DB_DIR = os.getenv("CHROMA_DIR", "chroma_db")
MODE = os.getenv("MODE", "ollama")  # 'ollama' | 'together'
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vs = Chroma(persist_directory=DB_DIR, embedding_function=embed)
retriever = vs.as_retriever(search_kwargs={"k": 4})

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 한국어/영어 문서에 정통한 어시스턴트야. "
     "주어진 '컨텍스트'에 근거해 한국어로 간결하게 답변하고, "
     "출처(파일명/URL)도 함께 요약해줘. 근거가 모자라면 '모르겠다'고 말해."),
    ("human", "질문: {question}\n\n컨텍스트:\n{context}")
])

def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "local")
        snippet = d.page_content[:500].replace("\n", " ")
        lines.append(f"- ({src}) {snippet}")
    return "\n".join(lines)

# LLM 선택
if MODE == "together":
    # 모델 예: lgai/exaone-deep-32b
    llm = ChatTogether(model=os.getenv("TOGETHER_MODEL", "lgai/exaone-deep-32b"),
                       temperature=0.1)
else:
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b"),
                     temperature=0.1)

from langchain_core.runnables import RunnableParallel, RunnableLambda

chain = (
    RunnableParallel(context=retriever | RunnableLambda(format_docs),
                     question=lambda x: x["question"])
    | PROMPT
    | llm
    | StrOutputParser()
)

class Q(BaseModel):
    question: str

app = FastAPI(title="DevDesk-RAG")

@app.post("/chat")
def chat(q: Q):
    return {"answer": chain.invoke({"question": q.question})}

@app.get("/health")
def health():
    return {"status": "healthy", "mode": MODE, "model": os.getenv("OLLAMA_MODEL" if MODE == "ollama" else "TOGETHER_MODEL")}

@app.get("/")
def root():
    return {"message": "DevDesk-RAG API", "endpoints": ["/chat", "/health"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
