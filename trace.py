 # type: ignore
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from itertools import chain

load_dotenv()
openai_client = wrap_openai(OpenAI())

@traceable(run_type="tool", name="Retriever")
def retriever(query: str):
    results = ['John is very good person', 'John is very young', 'John like Peter', 'Pete is very young']
    return results

@traceable(run_type="llm", name="LLM")
def rag(question):
    docs = retriever(question)
    system_message = """Answer the users question using only the provided information below:
    {docs}""".format(docs="\n".join(docs))

    return openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content

result =rag(input('Ask the question: '))
print(result)
