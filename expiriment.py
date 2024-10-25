from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.schemas import Example, Run
from dotenv import load_dotenv

load_dotenv()

#Create new dataset and examples
client = Client()

dataset_name="Elementary Animal Questions"
example_inputs = [
("What is the largest mammal?", "The blue whale"),
("What do mammals and birds have in common?", "They are both warm-blooded"),
]

dataset = client.create_dataset(
  dataset_name=dataset_name, 
  description="Questions and answers about animal phylogenetics.",
)
client.create_examples(
  inputs=[{"question": input_prompt} for input_prompt, _ in example_inputs],
  outputs=[{"answer": output_answer} for _, output_answer in example_inputs],
  metadata=[{"source": "Wikipedia"} for _ in example_inputs],
  dataset_id=dataset.id,
)

#Setup LLM
prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a concise chatbot. You never answer in more than 20 words."),
  ("user", "{question}")
])

chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0) | StrOutputParser()

#Setup Evaluator 
def is_concise_enough(root_run: Run, example: Example) -> dict:
    # Example is a single example from your dataset that your app is currently evaluating
    # Run contains information of a single call to your application using the example's input
    #
    # For more info on evaluators, see: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    score = len(root_run.outputs["output"]) < 3 * len(example.outputs["answer"])
    return {"key": "is_concise", "score": int(score)}

evaluate(
    chain.invoke,
    data=dataset_name,
    evaluators=[is_concise_enough],
    experiment_prefix="my first experiment "
)
