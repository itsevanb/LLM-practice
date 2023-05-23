from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
import os
import openai

# Access the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise Exception("Missing OPENAI_API_KEY environment variable")

# Set the API key for OpenAI
openai.api_key = api_key

# Set up the first chain, which will generate an outline
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write me an outline on {topic}",
)
llm = OpenAI(temperature=0.9, max_tokens=-1)
chain = LLMChain(llm=llm, prompt=prompt)

# Set up the second chain, which will generate a blog article based on the outline
second_prompt = PromptTemplate(
    input_variables=["outline"],
    template="""Write a blog article in the format of the given outline 

    Outline:
    {outline}""",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Combine the chains into a SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying the input variable for the first chain.
catchphrase = overall_chain.run("AI")
print(catchphrase)
