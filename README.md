# 🛠️ Machine Event Predictor

An interactive **Streamlit** app that simulates and showcases a predictive-maintenance
model for detecting **rare machine events** from daily fleet telemetry.

The model and methodology come from [`predictive_maintenance.ipynb`](predictive_maintenance.ipynb)
and the accompanying case presentation: an **extreme class-imbalance** (~1 : 1173) binary
classification problem solved with temporal feature engineering, an **Isolation Forest**
anomaly score, and a **cost-sensitive XGBoost** classifier with a tuned decision threshold.

---

## What's inside

| Page | What it does |
|------|--------------|
| **🏠 Home** | Frames the business problem, dataset, modelling approach, and headline metrics. |
| **📊 Data Explorer** | Class imbalance, feature distributions (event vs. normal), event timeline, per-machine drill-down. |
| **🔮 Event Simulator** | Dial in (or load) a machine's daily telemetry and see the live event probability, anomaly score, and risk band. |
| **🚨 Fleet Monitoring** | Score the whole fleet, rank machines by risk, and tune the alert threshold to balance recall vs. false alarms. |
| **📈 Model Performance** | Interactive threshold tuning, confusion matrix, precision–recall & ROC curves, feature importance. |

---

## Project layout

```
.
├── app.py                  # Streamlit entry point (Home page)
├── pages/                  # Streamlit multipage app
│   ├── 1_Data_Explorer.py
│   ├── 2_Event_Simulator.py
│   ├── 3_Fleet_Monitoring.py
│   └── 4_Model_Performance.py
├── pdm/                    # Core package (shared by training + app)
│   ├── config.py           # Paths, feature definitions, hyperparameters
│   ├── data.py             # Loading + feature engineering
│   ├── model.py            # Train / persist / inference (Iso Forest + XGBoost)
│   └── streamlit_utils.py  # Cached loaders & UI helpers
├── train.py                # Reproduces the notebook pipeline, writes models/
├── dataset/machine_event.csv
├── models/                 # Generated artifacts (gitignored)
└── requirements.txt
```

---

## Setup & run

The app was developed in a dedicated **conda** environment named `machine-event`.

```bash
# 1. Create and activate the environment
conda create -n machine-event python=3.11 -y
conda activate machine-event

# 2. Install dependencies
pip install -r requirements.txt
# XGBoost needs the OpenMP runtime on macOS:
conda install -c conda-forge llvm-openmp -y

# 3. Train the model (writes artifacts to models/)
python train.py
#   python train.py --retune   # optionally re-search the F1-optimal threshold

# 4. Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

---

## How the pipeline works

```
raw telemetry (feature1–9)  ─┐
date → day_of_week/day_of_month/month ─┼─► Isolation Forest ─► iso_score ─┐
                                       │                                  ├─► XGBoost ─► P(event)
                                       └──────────────────────────────────┘
                                                                            P(event) ≥ threshold ⇒ alert
```

- **Imbalance** is handled with `scale_pos_weight` (cost-sensitive learning), not resampling.
- **Average Precision** is the headline metric — accuracy/ROC-AUC are misleading when the
  positive class is <0.1% of the data.
- The **decision threshold** (default `0.022`) is tuned far below 0.5 to trade off the high
  cost of missed events (false negatives) against unnecessary maintenance (false positives).

All hyperparameters live in [`pdm/config.py`](pdm/config.py) and match the notebook's
Grid Search CV results.
