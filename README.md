## 🌌 Immersive Heuristic Reinforcement Learning (IHRL)

**Event-driven, adaptive, and explainable AI for real-time decision-making across domains.**

>  built with love by Uniq Games / The ASK Company Ltd.*

---

### 📂 Project Overview

The **IHRL** system is a fully modular, domain-agnostic reinforcement learning framework designed to process event streams in real time, learn through hybrid training (live + batch), and explain its decisions with natural language—all with analyst-in-the-loop support!

---

### 💡 Key Features

| Module                      | Description                                                |
| --------------------------- | ---------------------------------------------------------- |
| 🧠 **Big Data Ready**       | Ingests event streams (e.g., Google Trends, reports)       |
| 🌍 **Domain Agnostic**      | Easily adaptable across industries                         |
| 🕒 **Async Time Behavior**  | Supports asynchronous event-response + optional simulation |
| 🔄 **Event Mapping (DTE)**  | Event transformation + rule modeling with sliding windows  |
| 🧱 **Simulation Layer**     | Abstract/stateful simulations (non-spatial)                |
| 🧩 **Heuristic System**     | Semi-adaptive signals using DSL-config + learned patterns  |
| 🤖 **Agent Architecture**   | One agent per stream; uses DQN + hybrid training           |
| 🪄 **Reward Design**        | Custom state-action-reward shaping                         |
| 🧼 **Pre-processing**       | Normalization-only pipeline                                |
| 🔮 **Model Inference**      | Built on PyTorch                                           |
| 🔁 **Feedback Loop**        | Daemon scripts for continuous learning                     |
| 📊 **Dashboard**            | Streamlit-based local live dashboard                       |
| 💬 **LLM Explanations**     | Mistral + Ollama chat for decision transparency            |
| 📏 **Metrics Tracking**     | Accuracy, novelty, timeliness, real-world impact           |
| 🧪 **Baselines**            | Compare with RL & rule-based systems                       |
| ⚖️ **Bias Mitigation**      | Fairness, counterfactuals, distributional checks           |
| 👩‍💻 **Human-in-the-Loop** | Analyst review + retraining support                        |
| 🕵️ **Anonymization**       | Pre-ingestion masking options for privacy compliance       |

---

### 🧾 Main Scripts

| Script                    | Role                                                                   |
| ------------------------- | ---------------------------------------------------------------------- |
| `agent_trainer_daemon.py` | Core daemon to train agents with live + batch data                     |
| `live_dashboard.py`       | Streamlit-powered UI to visualize performance + metrics                |
| `ollama_chat.py`          | LLM-assisted chatbot using **Mistral via Ollama** to explain decisions |

---

### ⚙️ Tech Stack

* 🐍 Python 3.10+
* 🔥 PyTorch
* 🎯 Streamlit
* 🧠 Mistral via [Ollama](https://ollama.com/)
* 📚 NumPy, Pandas, Scikit-learn
* ☁️ Easily Dockerizable (optional)

---

### 🚀 Getting Started

#### 1. Clone the Repo

```bash
git clone https://github.com/MTS-Krishna/Immersive-Heuristic-Reinforcement-Learning.git
cd Immersive-Heuristic-Reinforcement-Learning
```

#### 2. Install Requirements

#### 3. Run Training Daemon

```bash
python agent_trainer_daemon.py
```

#### 4. Launch the Live Dashboard

```bash
streamlit run live_dashboard.py
```

> 💡 Tip: Make sure Ollama is running and Mistral is available!

---

### 🛡️ Security & Privacy

* Supports **event-level anonymization**
* Compliant with **data masking policies**
* Designed with **auditability** in mind

---

### 🫶 Built With Love

Crafted by **The ASK Company Ltd.**
