from flask import Flask, request, jsonify
# from zhipuai import ZhipuAI
import json
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional
from langchain_core.messages.ai import AIMessage
from langchain import PromptTemplate
# from langchain import LLMChain

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

from modelscope import snapshot_download
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_openai import ChatOpenAI
import openai
openai_api_key = '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'
openai_api_base = 'https://openrouter.ai/api/v1'
llm = ChatOpenAI(
    openai_api_key = openai_api_key,
    openai_api_base= openai_api_base,
    temperature= 0.2,
    
    model_name ='openai/gpt-4-turbo-preview',
    model_kwargs={
        'extra_body':{
            'top_p':0.1,
            'reprtition_penalty':1,
            'stop_token_ids':[2]
        }
    }
)

# llm.invoke('8只兔子有多少条腿？')
loader = TextLoader("FAQ.txt",encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5", revision='master')
embedding_path=model_dir
embeddings = HuggingFaceBgeEmbeddings(model_name = embedding_path)

vectorstore = FAISS.from_documents(
    docs,
    embedding= embeddings
)
retriever = vectorstore.as_retriever()

template = """根据以下内容回答问题:
{context}

问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
# model = ChatOpenAI()
output_parser = StrOutputParser()

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

# retrieval_chain.invoke("订单取消或未全量收货，但被订单占用的PW预算或合同额度没有被释放，该如何处理？")



# app = Flask(__name__)
# def qa_system(question):
#     # 你的问答系统
#     answer = retrieval_chain.invoke(question)
#     # answer = "这是回答"  # 这里暂定为固定答案
#     return answer
# @app.route('/', methods=['POST'])
# def api():
#     question = request.json.get('question')
#     answer = qa_system(question)
#     return jsonify(answer=answer)
# if __name__ == '__main__':
#     app.run(host='0.0.0.0')
# #############################################################################   
# from flask import Flask, request, jsonify
# app = Flask(__name__)
# def qa_system(question):
#     # 你的问答系统
#     answer = retrieval_chain.invoke(question)
#     # answer = "这是回答"  # 这里暂定为固定答案
#     return answer
# @app.route('/', methods=['GET'])
# def home():
#     return "欢迎来到我的问答系统，请通过POST请求向我提问。"
# @app.route('/', methods=['POST'])
# def api():
#     question = request.json.get('question')
#     answer = qa_system(question)
#     return jsonify(answer=answer)
# if __name__ == '__main__':
#     app.run(host='0.0.0.0')
###################################################################
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
class Question(BaseModel):
    question: str
app = FastAPI()
def qa_system(question):
    # 你的问答系统
    answer = retrieval_chain.invoke(question)
    # answer = "这是回答"  # 这里暂定为固定答案
    return answer
@app.post("/")
def api(question: Question):
    answer = qa_system(question.question)
    return {"answer": answer}

# uvicorn main:app --host=0.0.0.0 --port=8000