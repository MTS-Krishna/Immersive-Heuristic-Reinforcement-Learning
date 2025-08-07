import streamlit as st
import psutil
import time
import random
import numpy as np
import pandas as pd
from agent_trainer_daemon import DeepIHRLAgent
import torch
import unicodedata
import os
from collections import deque

from ollama_chat import ask_ollama

# --- Helper for Clean Text ---
def clean_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# --- Historical Data Load ---
history_df = pd.read_csv("agent_logs.csv") if os.path.exists("agent_logs.csv") else None

# --- Heuristic Rule ---
def heuristic_rule(cpu, mem):
    if cpu > 85 and mem > 80:
        return 1
    elif cpu > 65:
        return 0
    else:
        return 2

# --- Load Agent ---
agent = DeepIHRLAgent(state_size=3, action_size=3)
agent.load()
agent.load_params()
agent.load_memory()
agent.epsilon = 0.05

# --- Streamlit Setup ---
st.set_page_config("IHRL v2.0 Dashboard", layout="centered")
st.title("🧠 IHRL Agent - Live Reinforcement & Monitoring")

# --- Session State Init ---
if "running" not in st.session_state:
    st.session_state.running = False
if "logs" not in st.session_state:
    st.session_state.logs = []
if "accuracy" not in st.session_state:
    st.session_state.accuracy = {"total": 0, "correct": 0, 0: [0, 0], 1: [0, 0], 2: [0, 0]}
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {"cpu": 0.0, "mem": 0.0, "proc": 0, "action": -1, "expected": -1, "confidence": 0.0}
if "recent_states" not in st.session_state:
    st.session_state.recent_states = deque(maxlen=10)

# --- Action Labels ---
action_labels = {
    0: ("🟡 Moderate Attention", "#facc15"),
    1: ("🔴 Urgent Alert", "#ef4444"),
    2: ("🟢 Normal", "#22c55e")
}

# --- Chat Assistant ---
with st.expander("💬 Ask the IHRL Assistant", expanded=True):
    use_history = st.checkbox("📁 Include Historical Logs", value=True)
    question = st.text_input("Ask a question about the agent, system, or logs:")

    if st.button("Ask"):
        metrics = st.session_state.last_metrics
        cpu = metrics["cpu"]
        mem = metrics["mem"]
        proc = metrics["proc"]
        action = metrics["action"]
        expected = metrics["expected"]
        confidence = metrics["confidence"]

        context = f"Agent Accuracy: {st.session_state.accuracy['correct']} / {st.session_state.accuracy['total']}\n"
        for a in [0, 1, 2]:
            c, t = st.session_state.accuracy[a]
            context += f"Class {a}: {c}/{t} correct\n"

        context += "\nRecent Logs:\n"
        for log in st.session_state.logs[:10]:
            context += f"- {clean_text(log)}\n"

        context += f"\nCurrent System Metrics:\nCPU: {cpu:.1f}%, Memory: {mem:.1f}%, Processes: {proc}\n"
        context += f"Agent Prediction: {action} (Confidence: {confidence*100:.2f}%)\n"
        context += f"Heuristic Expectation: {expected}\n"

        if use_history and history_df is not None:
            context += "\nHistorical Agent Logs:\n"
            for _, row in history_df.tail(20).iterrows():
                line = f"Time: {row['timestamp']:.2f}, CPU: {row['cpu']}%, Mem: {row['mem']}%, Proc: {row['proc']}, Agent: {row['agent_action']}, Heuristic: {row['expected']}, Correct: {bool(row['correct'])}, Confidence: {row['confidence']:.2f}"
                context += f"- {clean_text(line)}\n"

        final_prompt = (
            f"You are an intelligent monitoring assistant. Only answer based on the context below. "
            f"Provide precise answers without guessing.\n\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = ask_ollama(final_prompt)
        st.success(answer)

# --- Buttons ---
start_btn = st.button("▶️ Start Monitoring")
stop_btn = st.button("⏹️ Stop Monitoring")
clear_btn = st.button("🧹 Clear Logs")

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if clear_btn:
    st.session_state.logs = []
    st.session_state.accuracy = {"total": 0, "correct": 0, 0: [0, 0], 1: [0, 0], 2: [0, 0]}

# --- UI Placeholders ---
status_box = st.empty()
metrics_box = st.empty()
accuracy_box = st.empty()
per_class_box = st.empty()
qval_box = st.empty()
log_box = st.empty()
log_data = []

# --- Live Monitoring Loop ---
while st.session_state.running:
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory().percent
    proc = len(psutil.pids())
    state = np.array([cpu, mem, proc])

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.model(state_tensor)
        probs = torch.softmax(q_values, dim=1)
    action = torch.argmax(q_values).item()
    confidence = probs[0][action].item()

    expected = heuristic_rule(cpu, mem)
    correct = (action == expected)

    # Reward
    st.session_state.recent_states.append((cpu, mem, action))
    stable_count = sum(1 for c, m, a in st.session_state.recent_states if c < 70 and m < 75)
    stability_bonus = (stable_count / len(st.session_state.recent_states)) if st.session_state.recent_states else 0

    cpu_penalty = max(0, cpu - 85) * 0.05
    mem_penalty = max(0, mem - 80) * 0.05
    bonus = 1.0 if cpu < 65 and mem < 70 else 0.0
    reward = bonus - (cpu_penalty + mem_penalty)
    reward += 0.5 if correct else -0.2
    reward += stability_bonus

    agent.remember(state, action, reward, state, True)
    agent.replay(batch_size=32)
    agent.update_target_model()

    # Accuracy Update
    st.session_state.accuracy["total"] += 1
    if correct:
        st.session_state.accuracy["correct"] += 1
    st.session_state.accuracy[expected][1] += 1
    if correct:
        st.session_state.accuracy[expected][0] += 1

    # Update last metrics
    st.session_state.last_metrics = {
        "cpu": cpu, "mem": mem, "proc": proc,
        "action": action, "expected": expected,
        "confidence": confidence
    }

    # UI Output
    label, color = action_labels[action]
    verdict = "✅ Correct" if correct else "❌ Wrong"
    status_box.markdown(f"""<div style='padding:1rem; background-color:{color}; color:black; border-radius:12px; text-align:center;'>
        <h3>{label}</h3><p>{verdict} - Confidence: {confidence*100:.1f}%</p></div>""", unsafe_allow_html=True)

    metrics_box.markdown(f"""
        **CPU:** {cpu:.1f}%  
        **Memory:** {mem:.1f}%  
        **Processes:** {proc}  
        **Agent Action Index:** `{action}`  
        **Heuristic Expectation:** `{expected}`
    """)

    log_msg = f"CPU: {cpu:.1f}%, Mem: {mem:.1f}%, Proc: {proc} → Agent: {label} | Heuristic: {expected} {verdict}"
    st.session_state.logs.insert(0, log_msg)
    st.session_state.logs = st.session_state.logs[:10]

    overall_acc = (st.session_state.accuracy["correct"] / st.session_state.accuracy["total"]) * 100
    accuracy_box.markdown(f"### 🎯 Overall Accuracy: **{overall_acc:.2f}%** over {st.session_state.accuracy['total']} predictions")

    pc = st.session_state.accuracy
    pc_display = "#### 📊 Per-Class Accuracy\n"
    for a in [0, 1, 2]:
        c, t = pc[a]
        val = (c / t * 100) if t else 0
        label, _ = action_labels[a]
        pc_display += f"- {label}: **{val:.2f}%** ({c}/{t})\n"
    per_class_box.markdown(pc_display)

    q_display = "#### 🧠 Q-Values & Probabilities\n"
    for i in range(3):
        label, _ = action_labels[i]
        q_display += f"- {label}: Q = `{q_values[0][i]:.3f}` | Prob = `{probs[0][i]*100:.2f}%`\n"
    qval_box.markdown(q_display)

    log_box.markdown("### 📜 Recent Logs:\n" + "\n".join([f"- {l}" for l in st.session_state.logs]))

    log_data.append([time.time(), cpu, mem, proc, action, expected, correct, confidence])
    time.sleep(1)

# --- Save Logs on Exit ---
if log_data:
    df_log = pd.DataFrame(log_data, columns=["timestamp", "cpu", "mem", "proc", "agent_action", "expected", "correct", "confidence"])
    df_log.to_csv("agent_logs.csv", index=False)