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

def validation_node(state: GraphState):
    print("Agent: Checking Conditions (Problem Statement, Dataset, ML Context)...")
    uid, sender, subject = tools.find_and_validate_email()
    if not uid:
        return {"error": "Requirements not met. Agent staying idle."}
    return {"email_uid": uid, "recipient": sender, "subject": subject, "error": None}

def ingest_node(state: GraphState):
    if state.get("error"): return state
    print(f"Agent: Conditions Met! Processing Project: {state['subject']}")
    df = tools.download_dataset_by_uid(state['email_uid'])
    
    # Send Initial Report
    msg = f"Project Confirmed.\nSubject: {state['subject']}\nDataset rows: {len(df)}\nStatus: Training Model..."
    tools.send_email_report(state['recipient'], "Agent Notification: Initial ML Report", msg)
    
    return {"dataset": df}

def ml_process_node(state: GraphState):
    if state.get("error"): return state
    print("Agent: Training ML Model and generating final results...")
    
    summary, score = run_manual_ml(state['dataset'])
    pdf = tools.generate_visualizations(state['dataset'])
    
    # Send Final Model Report
    final_body = f"Final ML Project Report\nAccuracy: {score:.4f}\n\nSummary:\n{summary}"
    tools.send_email_report(state['recipient'], "Agent Notification: Final Model Report", final_body, pdf)
    
    return {"accuracy": score}

# Define Graph
builder = StateGraph(GraphState)
builder.add_node("validate", validation_node)
builder.add_node("ingest", ingest_node)
builder.add_node("ml_process", ml_process_node)

builder.set_entry_point("validate")
builder.add_edge("validate", "ingest")
builder.add_edge("ingest", "ml_process")
builder.add_edge("ml_process", END)

app = builder.compile()

if __name__ == "__main__":
    result = app.invoke({"accuracy": 0.0, "error": None})
    if result.get("error"):
        print(f"Finished: {result['error']}")
    else:
        print("Success: All reports sent.")
