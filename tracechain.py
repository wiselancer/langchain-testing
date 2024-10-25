from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question}\nContext: {context}")
])

model = ChatOpenAI(model="gpt-3.5-turbo").with_config({"tags": ["model-tag-sam"], "metadata": {"model-key-sam": "model-value-sam"}})
output_parser = StrOutputParser()

chain = (prompt | model | output_parser).with_config({"run_name": "Sam test run", "tags": ["config-tag"], "metadata": {"config-key": "config-value"}})

question = "Can you summarize this morning's meetings?"
context = "During this morning's meeting, we solved all world conflict."

chain.invoke({"question": question, "context": context}, {"tags": ["invoke-tag"], "metadata": {"invoke-key": "invoke-value"}})