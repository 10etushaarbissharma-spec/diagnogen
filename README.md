# DiagnoGen AI Healthcare Support System (Phase-II)

This repository contains the Phase-II implementation of the DiagnoGen healthcare support system, featuring a modular architecture with a FastAPI backend and an interactive Streamlit frontend dashboard.

---

## Key Features Implemented

1. **Symptom Input UI**:
   - Natural language symptom description area.
   - Live NLP keyword parsing (matching triggers like "fever", "cough", "headache", "fatigue" to automatically toggle options).
   - Core checkboxes for manual adjustments and validation controls.

2. **Prediction Dashboard**:
   - Prominent predicted disease card with live confidence percentage indicators.
   - Interactive probability distribution bar chart showing alternative candidate classes.
   - Dynamic clinical guidance block mapping preliminary care recommendations to the predicted disease.

3. **Emergency Alert UI**:
   - Active real-time safety scanner looking for critical symptoms (e.g., chest pain, shortness of breath, breathing difficulty) or severe combinations.
   - Displays flashing warning banners instructing users to contact emergency services immediately.

4. **Dual Execution Modes (API Gateway & Fallback)**:
   - Queries the FastAPI API gateway `/predict` endpoint for inference by default.
   - Automatically falls back to local machine learning inference (loading the pickled model directly) if the API server is offline, with clear connection status badges.

5. **Developer & Model Controls Panel**:
   - Collapsible developer expander to run model training/retraining and preview raw training dataset records in the browser.

---

## Quick Start

### 1. Install Dependencies
Ensure you have all the required Python packages installed:
```bash
pip install -r requirements.txt
```

### 2. Start the Backend API Gateway (Optional but Recommended)
Run the FastAPI REST server:
```bash
uvicorn backend.main:app --port 8000 --reload
```
*Note: If the server is offline, the Streamlit app will gracefully detect it and fallback to local model prediction.*

### 3. Start the Streamlit Web Interface
Launch the dashboard frontend in a new terminal window:
```bash
streamlit run frontend/app.py
```
Open the local URL displayed in the terminal (typically `http://localhost:8501`) to interact with the application.

---

## File Structure

- [app.py](file:///c:/Users/hptfb/Downloads/DiagnoGen/DiagnoGen/frontend/app.py): The Streamlit frontend dashboard containing the input UI, prediction displays, emergency warnings, and developer controls.
- [main.py](file:///c:/Users/hptfb/Downloads/DiagnoGen/DiagnoGen/backend/main.py): FastAPI backend script exposing endpoints to compute prediction classes, model confidence, and alternative disease probabilities.
- [disease_model.pkl](file:///c:/Users/hptfb/Downloads/DiagnoGen/DiagnoGen/backend/disease_model.pkl): Trained Random Forest classifier model file.
- [symptoms.csv](file:///c:/Users/hptfb/Downloads/DiagnoGen/DiagnoGen/backend/symptoms.csv): Core dataset containing symptom parameters and their mapped disease categories.
- [train_on_synthetic.py](file:///c:/Users/hptfb/Downloads/DiagnoGen/DiagnoGen/backend/train_on_synthetic.py): Script to load symptoms and train the classifier with validation reports.

---

## Medical Disclaimer
DiagnoGen is an AI-assisted healthcare prototype designed for decision support and educational guidance. It is **not** a replacement for professional clinical diagnosis, treatment, or medical advice. If you are experiencing severe or worsening symptoms, consult a qualified healthcare provider or contact emergency services immediately.
