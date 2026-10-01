import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"]
)

# Agent 1: Research Agent
def research_agent(question):
    prompt = f"""
    You are a Research Agent.
    Analyze the following question.
    Identify the important concepts, requirements,
    benefits and challenges.

    Question: {question}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Agent 2: Technical Agent
def technical_agent(question, research):
    prompt = f"""
    You are a Kubernetes Technical Architect.
    Based on the question and research below,
    propose a technical solution.

    Include:
    - Architecture
    - Kubernetes components
    - Deployment approach
    - Security and monitoring

    Question: {question}
    Research:
    {research}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Agent 3: Report Agent
def report_agent(question, research, technical):
    prompt = f"""
    You are a Technical Report Agent.
    Create a structured final report with:
    1. Executive summary
    2. Research findings
    3. Technical architecture
    4. Implementation steps
    5. Conclusion

    User question: {question}

    Research findings:
    {research}

    Technical solution:
    {technical}

    Do not invent unsupported facts.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Coordinator Agent
def coordinator_agent(question):
    print("\n[Coordinator] Starting workflow")

    print("\n[Research Agent] Working...")
    research = research_agent(question)

    print("\n[Technical Agent] Working...")
    technical = technical_agent(question, research)

    print("\n[Report Agent] Working...")
    report = report_agent(question, research, technical)

    return report


# Main program
if __name__ == "__main__":
    question = input("Enter your question: ")

    result = coordinator_agent(question)

    print("\n========== FINAL REPORT ==========")
    print(result)
