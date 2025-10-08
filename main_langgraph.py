import os
import re
from typing import TypedDict, Optional
from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

# Import tools and constants
from tools import (
    download_dataset_from_email, run_pycaret_auto_ml, send_client_email,
    TARGET_ACCURACY_MIN, TARGET_ACCURACY_MAX
)

# Load environment variables (secrets)
load_dotenv()

# --- 1. CONFIGURATION AND LLM SETUP ---
MODEL_NAME = "tinyllama" # Ensure this model is pulled in Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CLIENT_EMAIL_TARGET = os.getenv("CLIENT_EMAIL_TARGET")

# Initialize LLM for narrative generation and reasoning
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_HOST, temperature=0.1)


# --- 2. LANGGRAPH STATE DEFINITION ---
class GraphState(TypedDict):
    """The shared state passed between agents."""
    dataset: Optional[pd.DataFrame]
    ml_report: Optional[str]
    accuracy: Optional[float]
    eda_insights: Optional[str]
    rca_business_impact: Optional[str]
    workflow_output: Optional[str]
    error: Optional[str]


# --- 3. AGENT NODES (TASKS) ---

def ingest_data_node(state: GraphState) -> GraphState:
    """Agent: Data Ingestion & Preparation (Initial Step)"""
    try:
        # Use the tool to download the email attachment
        df = download_dataset_from_email(subject_filter='Client Dataset')
        # Check if the last column is suitable for ML target
        if df.shape[1] < 2:
            raise ValueError("Dataset has insufficient columns (less than 2).")
            
        return {"dataset": df, "error": None}
    except Exception as e:
        return {"error": f"Ingestion Agent failed: {e}"}

def generate_eda_node(state: GraphState) -> GraphState:
    """Agent: EDA & Insight Generation"""
    df = state.get("dataset")
    if df is None: return {"error": "No dataset for EDA."}

    print("Agent: EDA & Insight - Generating summary via LLM.")
    
    # LLM Call (Agent 1): Generate simple, client-friendly insights
    prompt = f"""
    Based only on the data summary:\n{df.head().to_markdown()}\n and the columns: {df.columns.tolist()}. 
    Generate a simple, non-technical, high-level **Insights** and **Conclusion** (max 5 lines) 
    that a client can understand. Focus on the distribution of key features and their possible relation to the target ({df.columns[-1]}).
    """
    eda_insights = llm.invoke(prompt).content
    
    return {"eda_insights": eda_insights, "error": None}

def run_automl_node(state: GraphState) -> GraphState:
    """Agent: PyCaret AutoML"""
    df = state.get("dataset")
    if df is None: return {"error": "No dataset for AutoML."}

    report = run_pycaret_auto_ml(df)
    
    # Parse the primary metric (assumes the report structure returns Best Accuracy)
    # This is brittle, but necessary to get the numeric value for the Orchestrator
    accuracy_match = re.search(r"Primary Metric \((.*?)\): (\d\.\d{4})", report)
    accuracy = float(accuracy_match.group(2)) if accuracy_match else None
    
    return {"ml_report": report, "accuracy": accuracy, "error": None}

def generate_rca_node(state: GraphState) -> GraphState:
    """Agent: Model Evaluation & RCA"""
    report = state.get("ml_report")
    accuracy = state.get("accuracy")
    if report is None or accuracy is None: return {"error": "No ML report for RCA."}

    print("Agent: RCA & Business Impact - Analyzing via LLM.")

    # LLM Call (Agent 2): Generate RCA and business impact analysis
    prompt = f"""
    Analyze the following ML report:\n{report}\n and the primary metric {accuracy:.4f}.
    Generate two short, simple paragraphs: 
    1. **Root Cause Analysis (RCA):** Why the model performed as it did (e.g., factors contributing to the score).
    2. **Business Impact/Solution:** What the prediction means for the client's business decisions (e.g., focusing resources based on feature importance).
    """
    rca_business_impact = llm.invoke(prompt).content
    
    return {"rca_business_impact": rca_business_impact, "error": None}

def orchestrator_node(state: GraphState) -> GraphState:
    """Agent: Orchestrator, Monitoring, Approval, and Communication (Final Decision)"""
    accuracy = state.get("accuracy")
    
    if accuracy is None:
        return {"error": "Orchestrator failed: Missing accuracy score."}

    # 1. Approval Logic
    if TARGET_ACCURACY_MIN <= accuracy <= TARGET_ACCURACY_MAX:
        status = "APPROVED"
    else:
        status = "REJECTED_LOW_ACCURACY"
        
    # 2. Email Generation (Using LLM outputs)
    if status == "APPROVED":
        subject = "SUCCESS: Comprehensive ML Analysis and Business Insights"
        body = f"""Dear Client,

We are pleased to report that our ML model has successfully been trained and validated with a performance metric of **{accuracy*100:.2f}%**. This is within our target range, and the model is robust.

---
### 1. High-Level Insights & Conclusion
{state.get("eda_insights")}

---
### 2. Root Cause Analysis & Business Impact
{state.get("rca_business_impact")}

---
### 3. Model Performance Report
The model metrics show strong predictive power. We are ready to move to implementation.

Best Regards,
Your Agentic AI Team
"""
    else: # REJECTED_LOW_ACCURACY
        subject = "URGENT: Request for More Data - Initial Model Accuracy Low"
        # The LLM output is used here to explain the lack of patterns
        body = f"""Dear Client,

Thank you for sending the dataset. After running our comprehensive AutoML process, the model achieved a metric of only **{accuracy*100:.2f}%**. This is below our minimum reliability target of {TARGET_ACCURACY_MIN*100:.0f}%.

**Our findings (RCA):**
{state.get("rca_business_impact")}

**Action Required:**
The model indicates that the current data is insufficient to capture reliable patterns. To build a highly accurate and reliable prediction model, we require a **larger and more diverse dataset**. Please reply to this email with additional data.

We look forward to completing the project successfully.

Best Regards,
Your Agentic AI Team
"""
    
    # 3. Communication
    email_sent = send_client_email(subject, body, CLIENT_EMAIL_TARGET)

    output_message = f"Email sent successfully (Status: {status})" if email_sent else "Email failed to send."
    
    return {"workflow_output": output_message, "error": None}


# --- 4. LANGGRAPH WORKFLOW SETUP ---
workflow = StateGraph(GraphState)

# Add Nodes (5 Agents implemented as nodes)
workflow.add_node("ingest_data", ingest_data_node)
workflow.add_node("generate_eda", generate_eda_node)
workflow.add_node("run_automl", run_automl_node)
workflow.add_node("generate_rca", generate_rca_node)
workflow.add_node("orchestrator", orchestrator_node) # Also acts as the monitoring agent

# Set the entry point and sequential flow
workflow.set_entry_point("ingest_data")
workflow.add_edge("ingest_data", "generate_eda")
workflow.add_edge("generate_eda", "run_automl")
workflow.add_edge("run_automl", "generate_rca")
workflow.add_edge("generate_rca", "orchestrator")

# End the graph after the final communication
workflow.add_edge("orchestrator", END)

# Compile the Graph
app = workflow.compile()

if __name__ == "__main__":
    if not CLIENT_EMAIL_TARGET:
        print("ERROR: CLIENT_EMAIL_TARGET is not set in environment variables.")
    else:
        print("Starting LangGraph Multi-Agent Workflow...")
        
        # Run the graph
        final_state = app.invoke({})

        print("\n--- Final Workflow Summary ---")
        print(f"Status: {final_state['workflow_output']}")
        print(f"Model Accuracy: {final_state.get('accuracy')}")
