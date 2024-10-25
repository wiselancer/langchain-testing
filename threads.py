from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import uuid
load_dotenv()

prompt = ChatPromptTemplate.from_messages([("placeholder", "{messages}")])
model = ChatOpenAI(model="gpt-4o-mini")

chain = prompt | model

messages = [
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="Paris"),
    HumanMessage(content="What is the capital of Germany?"),
]
config = {"run_name": "Chat with" + str(uuid.uuid4()), "metadata": {"conversation_id": str(uuid.uuid4())}}

reponse = chain.invoke({"messages": messages}, config=config)

messages = messages + [reponse, HumanMessage(content="What is the capital of Italy?")]

reponse = chain.invoke({"messages": messages}, config=config)