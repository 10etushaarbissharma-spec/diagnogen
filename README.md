
# DiagnoGen MVP (Phase-II - local demo)

This repository contains a minimal local demo of the DiagnoGen Phase-II machine learning pipeline.

Goals for this MVP
- Minimal: no API routes, no containerization, no generative AI.
- Demonstrate the ML path with hardcoded demo cases and a local runner.

Quick start
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the demo runner (trains model if missing):

```bash
python backend/demo.py
```

What the demo does
- Trains (or loads) a RandomForest model using `backend/symptoms.csv`.
- Runs a set of hardcoded symptom cases and prints predicted disease labels and a simple confidence estimate when available.

Optional: Run the Streamlit UI
1. Install `streamlit` (if you want the UI):

```bash
pip install streamlit
```

2. Launch the UI:

```bash
streamlit run frontend/app.py
```

The Streamlit UI is retained for future integration; in this MVP it can be used locally to interact with the model if `backend/disease_model.pkl` exists or is created by `backend/demo.py`.

Generating a larger synthetic dataset
If you want to create a larger synthetic dataset for development or testing, use the following short Python snippet. Save it as `backend/generate_synthetic.py` and run it, or run the snippet interactively.

```python
import pandas as pd
import random

labels = ["Flu","Cold","Allergy","Dengue","Migraine","Covid","Bronchitis","Typhoid"]
rows = []
for i in range(2000):
	fever = random.choice([0,1])
	cough = random.choice([0,1])
	headache = random.choice([0,1])
	fatigue = random.choice([0,1])
	# naive label sampling: pick label correlated with some symptoms
	if fever and cough and headache:
		disease = random.choice(["Flu","Covid","Dengue"])
	elif headache and not fever:
		disease = random.choice(["Migraine","Allergy"])
	elif cough and not headache:
		disease = random.choice(["Cold","Bronchitis"])
	else:
		disease = random.choice(labels)
	rows.append({"fever":fever,"cough":cough,"headache":headache,"fatigue":fatigue,"disease":disease})

df = pd.DataFrame(rows)
df.to_csv("backend/symptoms_synthetic.csv", index=False)
print("Wrote backend/symptoms_synthetic.csv (2000 rows)")
```

Use the synthetic CSV to train a model by modifying `backend/demo.py` or by running `backend/model.py` with the new CSV path.

File map
- `backend/demo.py`: standalone demo runner (trains/loads model, prints hardcoded cases)
- `backend/model.py`: original training script (writes `backend/disease_model.pkl`)
- `backend/symptoms.csv`: small example dataset included with the repo
- `frontend/app.py`: Streamlit UI retained for optional local interaction
- `requirements.txt`: minimal dependencies for the demo

Notes & disclaimer
- This project is a prototype for educational/demonstration purposes only. It is not medical advice and must not be used for diagnosis.
- The synthetic dataset generator above is intentionally simple; if you require medically realistic synthetic data, consult domain experts and appropriate data sources.
