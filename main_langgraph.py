import os
from typing import TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import tools
from ml_advanced import run_manual_ml

class GraphState(TypedDict):
    email_uid: str
    recipient: str
    dataset: Optional[pd.DataFrame]
    accuracy: float
    report_text: str
    error: Optional[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

def context_aware_search_node(state: GraphState):
    print("Agent: Analyzing inbox context...")
    uid, sender = tools.find_data_email()
    return {"email_uid": uid, "recipient": sender}

def ingestion_node(state: GraphState):
    print(f"Agent: Downloading dataset from email UID {state['email_uid']}...")
    df = tools.download_dataset_by_uid(state['email_uid'])
    # Send Initial Report
    initial_msg = f"Hello! I've detected your request. I am processing the dataset with {df.shape[0]} rows and {df.shape[1]} columns. I will send the model results shortly."
    tools.send_email_report(state['recipient'], "Initial Data Receipt Report", initial_msg)
    return {"dataset": df}

def ml_processing_node(state: GraphState):
    print("Agent: Training Models...")
    summary, score = run_manual_ml(state['dataset'])
    pdf_path = tools.generate_visualizations(state['dataset'])
    
    final_msg = f"Model Training Complete!\n\nMetrics: {summary}\nBest Score: {score:.4f}\n\nPlease find the visual analysis attached."
    tools.send_email_report(state['recipient'], "Final Model Analysis Report", final_msg, pdf_path)
    return {"accuracy": score, "report_text": summary}

# Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("search", context_aware_search_node)
workflow.add_node("ingest", ingestion_node)
workflow.add_node("ml", ml_processing_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "ingest")
workflow.add_edge("ingest", "ml")
workflow.add_edge("ml", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"email_uid": "", "recipient": "", "accuracy": 0.0})
