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
    print("Agent: Validating ML Project Conditions...")
    uid, sender, subject = tools.find_and_validate_email()
    if not uid:
        return {"error": "Criteria not met (Problem Statement/Dataset/ML context missing)."}
    return {"email_uid": uid, "recipient": sender, "subject": subject}

def ingest_node(state: GraphState):
    if state.get("error"): return state
    print(f"Agent: Conditions Met. Ingesting {state['subject']}...")
    df = tools.download_dataset_by_uid(state['email_uid'])
    tools.send_email_report(state['recipient'], "Initial Report: Started", f"Processing {len(df)} rows.")
    return {"dataset": df}

def ml_process_node(state: GraphState):
    if state.get("error"): return state
    print("Agent: Building Model and Sending Final Report...")
    summary, score = run_manual_ml(state['dataset'])
    pdf = tools.generate_visualizations(state['dataset'])
    tools.send_email_report(state['recipient'], "Final Model Report", f"Accuracy: {score}\n{summary}", pdf)
    return {"accuracy": score}

# Building the workflow
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
    app.invoke({"accuracy": 0.0, "error": None})
