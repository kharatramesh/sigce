from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Create LLM
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)
# Create prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are an IT trainer.
    Explain the following topic to a beginner:
    Topic: {topic}
    Provide:
    1. Definition
    2. Real-world example
    3. Technical example
    4. Key points
    """
)

# Output parser
parser = StrOutputParser()

# Create chain
chain = prompt | llm | parser

# Execute chain
result = chain.invoke({
    "topic": "Kubernetes"
})

print(result)
