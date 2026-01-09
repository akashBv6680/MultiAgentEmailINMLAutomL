import os
import traceback 
from typing import TypedDict, Optional
from dotenv import load_dotenv
import pandas as pd
from langgraph.graph import StateGraph, END

# Use the latest Google Generative AI integration for LangChain
from langchain_google_genai import ChatGoogleGenerativeAI 

# --- Import the tools module (ensure tools.py is in the same directory) ---
import tools 
# -------------------------------

load_dotenv()

# --- 1. CONFIGURATION AND LLM SETUP ---
# Fetching the Gemini API Key from GitHub Secrets (env var: GOOGLE_API_KEY)
# We use gemini-2.5-flash as requested.
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise ValueError("CRITICAL: GOOGLE_API_KEY not found. Please add it to GitHub Secrets.")

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME, 
    google_api_key=GEMINI_API_KEY, 
    temperature=0.1,
    max_retries=2
)

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
    """Agent: Data Ingestion & Preparation"""
    try:
        print("Agent: Data Ingestion - Downloading dataset...")
        df = tools.download_dataset_from_email() 
        
        if df is None or df.shape[1] < 2:
            raise ValueError("Dataset is empty or has insufficient columns.")
            
        return {"dataset": df, "error": None}
    except Exception as e:
        full_trace = traceback.format_exc()
        return {"error": f"Ingestion Agent failed: {e}\n{full_trace}"}

def generate_eda_node(state: GraphState) -> GraphState:
    """Agent: EDA & Insight Generation (Uses Gemini 2.5 Flash)"""
    df = state.get("dataset")
    if df is None: return {"error": "No dataset found for EDA."}

    print(f"Agent: EDA - Generating insights using {LLM_MODEL_NAME}...")
    
    try:
        # Provide a snippet of the data to the LLM
        data_summary = df.head().to_string()
        columns = df.columns.tolist()
        
        prompt = f"""
        Analyze this data sample:
        {data_summary}
        
        Columns: {columns}
        
        Task: Generate a high-level summary (max 5 lines). 
        Identify the likely target variable ({columns[-1]}) and how other features might relate to it.
        Keep it non-technical for a business client.
        """
        response = llm.invoke(prompt)
        return {"eda_insights": response.content, "error": None}
    except Exception as e:
        return {"error": f"EDA Agent failed: {e}"} 

def run_automl_node(state: GraphState) -> GraphState:
    """Agent: Scikit-learn Training & Report Generation"""
    df = state.get("dataset")
    if df is None: return {"error": "No dataset for ML training."}
    
    try:
        print("Agent: ML Training - Running Scikit-Learn pipeline...")
        report, accuracy = tools.run_manual_ml(df)
        
        if accuracy is None:
            return {"error": f"ML Training Agent failed: {report}"}

        # Save visualizations locally (e.g., visual_report.pdf)
        tools.generate_visualizations(df)
        
        return {"ml_report": report, "accuracy": accuracy, "error": None}
    except Exception as e:
        return {"error": f"ML Training Agent failed: {e}"}

def generate_rca_node(state: GraphState) -> GraphState:
    """Agent: Model Evaluation & RCA (Uses Gemini 2.5 Flash)"""
    report = state.get("ml_report")
    accuracy = state.get("accuracy")
    if report is None: return {"error": "No ML report found for RCA."}

    print(f"Agent: RCA - Analyzing model performance with {LLM_MODEL_NAME}...")

    try:
        prompt = f"""
        ML Performance Report:
        {report}
        
        Accuracy (R2 Score): {accuracy:.4f}
        
        Task:
        1. Root Cause Analysis (RCA): Why did the model achieve this score?
        2. Business Impact: What should the client do next based on these results?
        Keep the response concise and professional.
        """
        response = llm.invoke(prompt)
        return {"rca_business_impact": response.content, "error": None}
    except Exception as e:
        return {"error": f"RCA Agent failed: {e}"}

def orchestrator_node(state: GraphState) -> GraphState:
    """Agent: Final Decision & Workflow Summary"""
    accuracy = state.get("accuracy")
    eda = state.get("eda_insights")
    rca = state.get("rca_business_impact")
    
    if accuracy is None:
        return {"error": "Orchestrator failed: Missing accuracy data."}

    # Business Logic Thresholds
    status = "APPROVED" if accuracy >= tools.TARGET_ACCURACY_MIN else "REJECTED_LOW_CONFIDENCE"
    
    # Consolidate everything into a final output string
    final_summary = f"""
    --- AGENTIC WORKFLOW FINAL REPORT ---
    STATUS: {status}
    MODEL ACCURACY: {accuracy:.4f}
    
    [EDA INSIGHTS]
    {eda}
    
    [BUSINESS IMPACT & RCA]
    {rca}
    -------------------------------------
    """
    print("Agent: Orchestrator - Workflow complete.")
    return {"workflow_output": final_summary, "error": None}


# --- 4. LANGGRAPH WORKFLOW SETUP ---
workflow = StateGraph(GraphState)

# Define Nodes
workflow.add_node("ingest_data", ingest_data_node)
workflow.add_node("generate_eda", generate_eda_node)
workflow.add_node("run_automl", run_automl_node)
workflow.add_node("generate_rca", generate_rca_node)
workflow.add_node("orchestrator", orchestrator_node)

# Define Edges (Linear Flow)
workflow.set_entry_point("ingest_data")
workflow.add_edge("ingest_data", "generate_eda")
workflow.add_edge("generate_eda", "run_automl")
workflow.add_edge("run_automl", "generate_rca")
workflow.add_edge("generate_rca", "orchestrator")
workflow.add_edge("orchestrator", END)

app = workflow.compile()

# --- 5. EXECUTION ---
if __name__ == "__main__":
    print(f"--- Starting Agentic ML Workflow ({LLM_MODEL_NAME}) ---")
    
    # Run the graph
    result = app.invoke({})

    if result.get("error"):
        print(f"\nWORKFLOW FAILED:\n{result['error']}")
    else:
        print(result["workflow_output"])
