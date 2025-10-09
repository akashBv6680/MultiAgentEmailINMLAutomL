import os
import re
import traceback 
from typing import TypedDict, Optional
from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama 

# --- Import the tools module ---
import tools 
# -------------------------------

load_dotenv()

# --- 1. CONFIGURATION AND LLM SETUP ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "tinyllama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CLIENT_EMAIL_TARGET = os.getenv("CLIENT_EMAIL_TARGET")

llm = ChatOllama(model=LLM_MODEL_NAME, base_url=OLLAMA_HOST, temperature=0.1)


# --- 2. LANGGRAPH STATE DEFINITION (Unchanged) ---
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
    """Agent: Data Ingestion & Preparation"""
    try:
        # Calls the function using module prefix (tools.) 
        df = tools.download_dataset_from_email() 
        
        if df.shape[1] < 2:
            raise ValueError("Dataset has insufficient columns (less than 2).")
        return {"dataset": df, "error": None}
    except Exception as e:
        print(f"Ingestion Agent caught exception: {e}")
        full_trace = traceback.format_exc()
        return {"error": f"Ingestion Agent failed: {e}\n\nFull Trace:\n{full_trace}"}

def generate_eda_node(state: GraphState) -> GraphState:
    """Agent: EDA & Insight Generation (Uses Ollama)"""
    df = state.get("dataset")
    if df is None: return {"error": "No dataset for EDA."}

    print("Agent: EDA & Insight - Generating summary via LLM.")
    
    try:
        # Removed .to_markdown() for compatibility, using .head() string representation
        prompt = f"""
        Based only on the data summary:\n{df.head().to_string()}\n and the columns: {df.columns.tolist()}. 
        Generate a simple, non-technical, high-level **Insights** and **Conclusion** (max 5 lines) 
        that a client can understand. Focus on the distribution of key features and their possible relation to the target ({df.columns[-1]}).
        """
        eda_insights = llm.invoke(prompt).content
        
        return {"eda_insights": eda_insights, "error": None}
    except Exception as e:
        return {"error": f"EDA Agent failed: LLM Call Error: {e}"} 

def run_automl_node(state: GraphState) -> GraphState:
    """Agent: Manual Scikit-learn Training & Report Generation"""
    df = state.get("dataset")
    if df is None: return {"error": "No dataset for ML training."}
    
    try:
        # 1. Run Manual ML Training (returns report string and R2 score)
        report, accuracy = tools.run_manual_ml(df)
        
        if accuracy is None:
            # Check if the manual function returned an error string
            return {"error": f"ML Training Agent failed: {report}"}

        # 2. Generate Visualizations (Saves file but doesn't return it)
        tools.generate_visualizations(df)
        
        return {"ml_report": report, "accuracy": accuracy, "error": None}
    
    except Exception as e:
        print(f"ML Training Agent caught unhandled exception: {e}")
        full_trace = traceback.format_exc()
        return {"error": f"ML Training Agent failed: Unhandled Exception: {e}\n\nFull Trace:\n{full_trace}"}


def generate_rca_node(state: GraphState) -> GraphState:
    """Agent: Model Evaluation & RCA (Uses Ollama)"""
    report = state.get("ml_report")
    accuracy = state.get("accuracy")
    if report is None or accuracy is None: return {"error": "No ML report for RCA."}

    print("Agent: RCA & Business Impact - Analyzing via LLM.")

    try:
        prompt = f"""
        Analyze the following ML report:\n{report}\n and the primary metric {accuracy:.4f} (R-squared score).
        Generate two short, simple paragraphs: 
        1. **Root Cause Analysis (RCA):** Why the model performed as it did (e.g., factors contributing to the score, model selection).
        2. **Business Impact/Solution:** What the prediction means for the client's business decisions (e.g., focusing resources based on feature importance).
        """
        rca_business_impact = llm.invoke(prompt).content
        
        return {"rca_business_impact": rca_business_impact, "error": None}
    except Exception as e:
        return {"error": f"RCA Agent failed: LLM Call Error: {e}"}


def orchestrator_node(state: GraphState) -> GraphState:
    """Agent: Orchestrator, Monitoring, Approval, and Communication (Final Decision)"""
    accuracy = state.get("accuracy")
    eda_insights = state.get("eda_insights")
    rca_business_impact = state.get("rca_business_impact")
    ml_report = state.get("ml_report")

    # This check is what was failing, but should now pass after manual ML
    if accuracy is None:
        return {"error": "Orchestrator failed: Missing accuracy score."}

    # Using tools. prefix for constants
    status = "APPROVED" if tools.TARGET_ACCURACY_MIN <= accuracy <= tools.TARGET_ACCURACY_MAX else "REJECTED_LOW_ACCURACY"
    
    # Ensure client target email is set
    if not CLIENT_EMAIL_TARGET:
        return {"error": "CLIENT_EMAIL_TARGET environment variable is not set."}
        
    # Email generation logic
    subject = ""
    if status == "APPROVED":
        subject = f"SUCCESS: ML Analysis Complete - High Confidence (R2: {accuracy:.4f})"
        body = f"""Dear Client,
We have successfully analyzed your data and completed the machine learning model training.

--- Summary ---
Status: APPROVED (R-squared Score: {accuracy:.4f})
{rca_business_impact}

--- Raw Report ---
{ml_report}

--- EDA Insights ---
{eda_insights}

A visual report (visual_report.pdf) was also generated locally.
"""
    else:
        subject = f"URGENT: Request for More Data - Low Model Confidence (R2: {accuracy:.4f})"
        body = f"""Dear Client,
The machine learning model training completed, but the confidence score (R-squared) of {accuracy:.4f} is below our minimum threshold of {tools.TARGET_ACCURACY_MIN:.2f}.

--- RCA Summary ---
{rca_business_impact}

To improve the model's performance, we strongly recommend:
1. Providing a larger dataset.
2. Including more relevant features/variables.

We await your feedback to proceed.
"""
    
    # Calling the function using module prefix (tools.)
    email_sent = tools.send_client_email(subject, body, CLIENT_EMAIL_TARGET)

    output_message = f"Email sent successfully (Status: {status})" if email_sent else "Email failed to send."
    
    return {"workflow_output": output_message, "error": None}


# --- 4. LANGGRAPH WORKFLOW SETUP (Unchanged) ---
workflow = StateGraph(GraphState)
workflow.add_node("ingest_data", ingest_data_node)
workflow.add_node("generate_eda", generate_eda_node)
workflow.add_node("run_automl", run_automl_node)
workflow.add_node("generate_rca", generate_rca_node)
workflow.add_node("orchestrator", orchestrator_node)
workflow.set_entry_point("ingest_data")
workflow.add_edge("ingest_data", "generate_eda")
workflow.add_edge("generate_eda", "run_automl")
workflow.add_edge("run_automl", "generate_rca")
workflow.add_edge("generate_rca", "orchestrator")
workflow.add_edge("orchestrator", END)
app = workflow.compile()

if __name__ == "__main__":
    if not CLIENT_EMAIL_TARGET:
        print("ERROR: CLIENT_EMAIL_TARGET is not set in environment variables.")
    else:
        print("Starting LangGraph Multi-Agent Workflow...")
        
        final_state = app.invoke({})

        print("\n--- Final Workflow Summary ---")
        
        if final_state.get('error'):
            print("Status: FAILED")
            print(f"Error Message: {final_state['error']}")
        else:
            print(f"Status: {final_state['workflow_output']}")
            print(f"Model Accuracy (R2): {final_state.get('accuracy')}")
