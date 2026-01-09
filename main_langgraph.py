import os
from typing import TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END
import tools
from ml_advanced import run_manual_ml

class GraphState(TypedDict):
    email_uid: Optional[str]
    recipient: Optional[str]
    subject: Optional[str]
    dataset: Optional[pd.DataFrame]
    accuracy: float
    error: Optional[str]

def validation_gate_node(state: GraphState):
    print("Agent: Checking for Problem Statement, Dataset, and ML Context...")
    uid, sender, subject = tools.find_and_validate_email()
    if not uid:
        return {"error": "No valid ML request found matching all conditions."}
    return {"email_uid": uid, "recipient": sender, "subject": subject}

def ingest_and_notify_node(state: GraphState):
    if state.get("error"): return state
    print(f"Agent: Conditions met. Downloading data from: {state['subject']}")
    
    df = tools.download_dataset_by_uid(state['email_uid'])
    
    # Initial Report
    report = f"Initial Report: Valid ML Project detected.\nSubject: {state['subject']}\nRows: {len(df)}\nStatus: Processing..."
    tools.send_email_report(state['recipient'], "Agent Receipt: Project Confirmed", report)
    
    return {"dataset": df}

def ml_and_final_report_node(state: GraphState):
    if state.get("error"): return state
    print("Agent: Running ML Pipeline...")
    
    summary, score = run_manual_ml(state['dataset'])
    pdf_path = tools.generate_visualizations(state['dataset'])
    
    # Final Report
    final_body = f"Final ML Report\nBest Model Accuracy: {score:.4f}\nSummary: {summary}"
    tools.send_email_report(state['recipient'], "Final ML Project Report", final_body, pdf_path)
    
    return {"accuracy": score}

# Define the Workflow
builder = StateGraph(GraphState)
builder.add_node("validate", validation_gate_node)
builder.add_node("ingest", ingest_and_notify_node)
builder.add_node("ml_process", ml_processing_node)

builder.set_entry_point("validate")
builder.add_edge("validate", "ingest")
builder.add_edge("ingest", "ml_process")
builder.add_edge("ml_process", END)

app = builder.compile()

if __name__ == "__main__":
    result = app.invoke({"accuracy": 0.0, "error": None})
    if result.get("error"):
        print(f"Agent Standby: {result['error']}")
    else:
        print("Agent Task Complete.")
