import os
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Import the PDF library
from markdown_pdf import MarkdownPdf, Section


# Initialize Groq LLM (Updated to a valid Groq model)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"]
)


# Shared state for all agents
class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str
    pdf_path: str  # <--- Added new state property


# Coordinator node
def coordinator(state: AgentState):
    print("\n[Coordinator] Starting workflow")
    return {
        "research": "",
        "technical": "",
        "report": "",
        "pdf_path": ""
    }


# Agent 1: Research Agent
def research_agent(state: AgentState):
    print("[Research Agent] Working...")

    prompt = f"""
    Analyze the following question as a Research Agent.

    Identify:
    - Important concepts
    - Requirements
    - Benefits
    - Challenges

    Question: {state['question']}
    """

    response = llm.invoke([
        SystemMessage(content="You are an IT Research Agent."),
        HumanMessage(content=prompt)
    ])

    return {"research": response.content}


# Agent 2: Technical Agent
def technical_agent(state: AgentState):
    print("[Technical Agent] Working...")

    prompt = f"""
    You are a Kubernetes Technical Architect.

    Based on the research, propose a technical solution.

    Include:
    - Architecture
    - Kubernetes components
    - Deployment approach
    - Security
    - Monitoring

    User question:
    {state['question']}

    Research findings:
    {state['research']}
    """

    response = llm.invoke([
        SystemMessage(content="You are a Kubernetes expert."),
        HumanMessage(content=prompt)
    ])

    return {"technical": response.content}


# Agent 3: Report Agent
def report_agent(state: AgentState):
    print("[Report Agent] Working...")

    prompt = f"""
    Prepare a structured technical report.

    Include:
    1. Executive summary
    2. Research findings
    3. Technical architecture
    4. Implementation steps
    5. Conclusion

    User question:
    {state['question']}

    Research:
    {state['research']}

    Technical solution:
    {state['technical']}

    Do not invent unsupported facts.
    """

    response = llm.invoke([
        SystemMessage(content="You are a Technical Report Agent."),
        HumanMessage(content=prompt)
    ])

    return {"report": response.content}


# Agent 4: PDF Generator Node
def pdf_generator(state: AgentState):
    print("[PDF Generator] Converting report to PDF...")
    
    file_name = "Technical_Report.pdf"
    
    try:
        # Create PDF document and add the markdown report as a section
        pdf = MarkdownPdf(toc_level=2) # toc_level automatically generates a Table of Contents
        pdf.add_section(Section(state['report']))
        
        # Save the file
        pdf.save(file_name)
        print(f"[PDF Generator] Successfully saved PDF as: {file_name}")
        
    except Exception as e:
        print(f"[PDF Generator] Error generating PDF: {e}")

    return {"pdf_path": file_name}


# Build LangGraph
graph = StateGraph(AgentState)

# Register nodes
graph.add_node("coordinator", coordinator)
graph.add_node("research", research_agent)
graph.add_node("technical", technical_agent)
graph.add_node("report", report_agent)
graph.add_node("pdf_generation", pdf_generator) # <--- Register new PDF node

# Define workflow
graph.add_edge(START, "coordinator")
graph.add_edge("coordinator", "research")
graph.add_edge("research", "technical")
graph.add_edge("technical", "report")
graph.add_edge("report", "pdf_generation") # <--- Route Report to PDF Generator
graph.add_edge("pdf_generation", END)      # <--- End workflow after PDF is created

# Compile graph
app = graph.compile()


# Execute application
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    question = input("Enter your question: ")

    result = app.invoke({
        "question": question,
        "research": "",
        "technical": "",
        "report": "",
        "pdf_path": ""
    })

    print("\n========== FINAL REPORT ==========")
    print(result["report"])
    print("\n==================================")
    print(f"📄 Report has been saved to: {os.path.abspath(result['pdf_path'])}")
