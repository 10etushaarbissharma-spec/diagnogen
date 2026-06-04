import streamlit as st
import pandas as pd
import os
import joblib
import requests
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import shap
import plotly.graph_objects as go

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "backend" / "disease_model.pkl"
SYN_PATH = ROOT / "backend" / "symptoms_synthetic.csv"
DATA_PATH = SYN_PATH if SYN_PATH.exists() else ROOT / "backend" / "symptoms.csv"
API_URL = "http://localhost:8000/predict"

# Page configuration
st.set_page_config(
    page_title="DiagnoGen - AI Healthcare Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Main container and theme */
    [data-testid="stAppViewContainer"] {
        background-color: #090D16;
        color: #E2E8F0;
    }
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Connection Status */
    .status-badge {
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 20px;
    }
    .status-online {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-offline {
        background-color: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Cards and boxes */
    .section-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Prediction block */
    .prediction-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #111827 100%);
        border: 2px solid #818CF8;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(129, 140, 248, 0.2);
        margin-bottom: 20px;
    }
    .disease-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38BDF8;
        margin: 10px 0;
    }
    .confidence-label {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 5px;
    }
    
    /* Emergency Alert box */
    .emergency-box {
        background-color: rgba(220, 38, 38, 0.1);
        border: 2px solid #DC2626;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .emergency-title {
        color: #EF4444;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .emergency-text {
        color: #FCA5A5;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for Model Training and Loading
def train_and_save_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path)
    X = df[["fever", "cough", "headache", "fatigue"]]
    y = df["disease"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save_model()

# Health guidance mapping based on disease prediction
def get_guidance(disease):
    guidance = {
        "Flu": "Rest, keep hydrated, and use OTC remedies for body aches. If symptoms worsen, consult a physician.",
        "Covid": "Isolate from others, monitor oxygen levels, rest, and keep hydrated. Seek emergency care for breathing difficulties.",
        "Cold": "Stay warm, drink plenty of fluids, and rest. Cough drops and nasal sprays may offer symptomatic relief.",
        "Allergy": "Identify and avoid allergens. Consider over-the-counter antihistamines or nasal sprays.",
        "Dengue": "Ensure bed rest and maintain optimal hydration. Avoid aspirin or ibuprofen (prefer acetaminophen). Watch for warning signs.",
        "Migraine": "Rest in a quiet, dark room. Apply a cool compress to your head. Avoid sensory triggers.",
        "Bronchitis": "Stay hydrated, use a humidifier, and avoid smoke or irritants. Seek care if coughing lasts over 3 weeks.",
        "Typhoid": "Requires clinical evaluation and antibiotic therapy. Ensure safe drinking water and soft nutrition."
    }
    return guidance.get(disease, "Rest and monitor your symptoms. Consult a doctor for diagnostic confirmation.")

# Backend API Status Check
def check_backend():
    try:
        r = requests.get("http://localhost:8000/", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False

# Predict Service Wrapper
def run_prediction(fever, cough, headache, fatigue, api_online):
    payload = {
        "fever": int(fever),
        "cough": int(cough),
        "headache": int(headache),
        "fatigue": int(fatigue)
    }
    
    if api_online:
        try:
            res = requests.post(API_URL, json=payload, timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                if "error" not in data:
                    return data["predicted_disease"], data["confidence"], data["probabilities"], data.get("shap_analysis")
        except Exception:
            pass
            
    # Local Fallback
    try:
        model = load_or_train_model()
        features = pd.DataFrame([[int(fever), int(cough), int(headache), int(fatigue)]], columns=["fever", "cough", "headache", "fatigue"])
        pred = model.predict(features)[0]
        confidence = 1.0
        probabilities = {}
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            classes = model.classes_
            probabilities = {cls: float(prob) for cls, prob in zip(classes, probs)}
            confidence = float(probabilities.get(pred, 1.0))
            
        shap_analysis = None
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(features)
            classes = list(model.classes_)
            pred_idx = classes.index(pred)
            
            feature_names = ["fever", "cough", "headache", "fatigue"]
            contributions = {
                name: float(shap_vals[0, idx, pred_idx])
                for idx, name in enumerate(feature_names)
            }
            
            base_value = float(explainer.expected_value[pred_idx])
            
            shap_analysis = {
                "base_value": base_value,
                "final_value": float(confidence),
                "contributions": contributions
            }
        except Exception:
            pass
            
        return pred, confidence, probabilities, shap_analysis
    except Exception as e:
        return f"Error: {str(e)}", 0.0, {}, None

# Check connection status
api_online = check_backend()

# Header layout
st.markdown("<div class='main-title'>🩺 DiagnoGen Support Center</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Clinical Decision Support & Disease Prediction Dashboard</div>", unsafe_allow_html=True)

if api_online:
    st.markdown("<span class='status-badge status-online'>🟢 Connected to DiagnoGen API Gateway</span>", unsafe_allow_html=True)
else:
    st.markdown("<span class='status-badge status-offline'>🟡 Offline - Running in Local Fallback Mode</span>", unsafe_allow_html=True)

# Layout: Split into Left and Right Columns
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("###  Symptom Input UI")
    
    # Text Area for NLP parser
    nlp_text = st.text_area(
        "Describe your symptoms in your own words:",
        value="",
        placeholder="e.g., I have a high fever, chest pain, and severe fatigue since yesterday...",
        help="Type a natural description. Keywords will automatically toggle checkboxes below."
    )
    
    # NLP keyword auto-matching
    nlp_text_lower = nlp_text.lower()
    auto_fever = any(kw in nlp_text_lower for kw in ["fever", "warm", "chills", "hot", "temperature"])
    auto_cough = any(kw in nlp_text_lower for kw in ["cough", "coughing", "throat", "sneeze", "sneezing"])
    auto_headache = any(kw in nlp_text_lower for kw in ["headache", "head pain", "migraine", "temple pain"])
    auto_fatigue = any(kw in nlp_text_lower for kw in ["fatigue", "tired", "weakness", "exhausted", "sleepy", "lethargic"])
    
    # Render switches / checkboxes pre-toggled by NLP results
    st.write("**Verify extracted symptoms:**")
    col_sym1, col_sym2 = st.columns(2)
    with col_sym1:
        fever = st.checkbox("Fever", value=auto_fever, help="Fever, chills, or hot flushes")
        cough = st.checkbox("Cough", value=auto_cough, help="Dry or productive cough, throat irritation")
    with col_sym2:
        headache = st.checkbox("Headache", value=auto_headache, help="Migraines, tension headaches, or head pain")
        fatigue = st.checkbox("Fatigue", value=auto_fatigue, help="Extreme tiredness, low energy, or weakness")
        
    st.write("")
    
    # Emergency Alert UI Triggering
    emergency_keywords = ["chest pain", "shortness of breath", "breathing difficulty", "difficulty breathing", "numbness", "paralysis", "confusion", "seizure", "unconscious"]
    has_emergency_keyword = any(kw in nlp_text_lower for kw in emergency_keywords)
    has_severe_combination = (fever and fatigue and cough) or (fever and cough and headache and fatigue)
    
    if nlp_text and (has_emergency_keyword or has_severe_combination):
        st.markdown(f"""
        <div class="emergency-box">
            <div class="emergency-title">⚠️ EMERGENCY ALERT: Critical Symptoms Detected</div>
            <div class="emergency-text">
                Your entry suggests potentially life-threatening or severe symptoms (e.g., respiratory distress, cardiac stress, or high-risk systemic combination). 
                <br><br>
                <b>Recommended Actions:</b>
                <ul>
                    <li>Please call <b>911</b> or contact emergency services immediately.</li>
                    <li>Go to the nearest emergency medical facility.</li>
                    <li>Do not wait for diagnostic predictions.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    st.markdown("###  Prediction Dashboard")
    
    has_symptoms = fever or cough or headache or fatigue
    
    if not has_symptoms:
        st.info("Enter symptoms on the left to view predictions, probabilities, and guidance.")
    else:
        # Run prediction
        with st.spinner("Processing prediction..."):
            disease, confidence, probabilities, shap_analysis = run_prediction(fever, cough, headache, fatigue, api_online)
            
        # Display Prediction Result Card
        st.markdown(f"""
        <div class="prediction-card">
            <div class="confidence-label">Primary Diagnostic Prediction</div>
            <div class="disease-title">{disease}</div>
            <div class="confidence-label">Prediction Confidence</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #818CF8; margin-top: 5px;">
                {confidence * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Confidence Progress Bar
        st.progress(confidence)
        
        # Display Alternative Probabilities Chart if probabilities dictionary is populated
        if probabilities:
            st.write("**Disease Probability Distribution:**")
            prob_df = pd.DataFrame(
                list(probabilities.values()),
                index=list(probabilities.keys()),
                columns=["Probability"]
            ).sort_values("Probability", ascending=True)
            
            st.bar_chart(prob_df, use_container_width=True)
            
        # Display SHAP explanation if available
        if shap_analysis:
            st.write("")
            st.write(f"**Symptom Contribution Analysis (SHAP) for {disease}:**")
            
            # Map features to present/absent states
            symptom_states = {
                "fever": "Present" if fever else "Absent",
                "cough": "Present" if cough else "Absent",
                "headache": "Present" if headache else "Absent",
                "fatigue": "Present" if fatigue else "Absent"
            }
            
            features_display = [
                f"Fever ({symptom_states['fever']})",
                f"Cough ({symptom_states['cough']})",
                f"Headache ({symptom_states['headache']})",
                f"Fatigue ({symptom_states['fatigue']})"
            ]
            
            contributions = shap_analysis["contributions"]
            values = [
                contributions.get("fever", 0.0),
                contributions.get("cough", 0.0),
                contributions.get("headache", 0.0),
                contributions.get("fatigue", 0.0)
            ]
            
            # Sort features by absolute contribution magnitude
            sorted_indices = sorted(range(len(values)), key=lambda k: abs(values[k]), reverse=True)
            features_display_sorted = [features_display[i] for i in sorted_indices]
            values_sorted = [values[i] for i in sorted_indices]
            
            # Determine color coding: light blue/teal for positive, rose/pink for negative
            colors = ["#38BDF8" if val >= 0 else "#F43F5E" for val in values_sorted]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=features_display_sorted,
                x=values_sorted,
                orientation='h',
                marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)', width=0)),
                text=[f"+{val*100:.1f}%" if val >= 0 else f"{val*100:.1f}%" for val in values_sorted],
                textposition='auto',
                hovertemplate="Symptom: %{y}<br>Contribution: %{x:+.1%}<extra></extra>"
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    title=dict(
                        text="Probability Shift (SHAP Value)",
                        font=dict(color='#94A3B8', size=11, family='Inter')
                    ),
                    tickfont=dict(color='#94A3B8', size=10, family='Inter'),
                    gridcolor='#1F2937',
                    zerolinecolor='#94A3B8',
                    zerolinewidth=1.5,
                    tickformat='+.0%'
                ),
                yaxis=dict(
                    tickfont=dict(color='#E2E8F0', size=11, family='Inter'),
                    autorange="reversed"
                ),
                height=220,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            base_pct = shap_analysis['base_value'] * 100
            final_pct = shap_analysis['final_value'] * 100
            st.markdown(f"""
            <div class="section-card" style="margin-top: 10px; border-left: 4px solid #818CF8; padding: 15px;">
                <h4 style="margin-top:0; color: #818CF8; font-size: 1.05rem;">💡 Interpretability Insight</h4>
                <p style="font-size: 0.9rem; line-height: 1.5; color: #CBD5E1; margin-bottom: 0;">
                    The model's base probability for <b>{disease}</b> is <b>{base_pct:.1f}%</b> (average rate in the training dataset). 
                    Your specific symptom combination shifts this prediction probability by <b>{(final_pct - base_pct):+.1f}%</b>, resulting in a final prediction confidence of <b>{final_pct:.1f}%</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Display AI Healthcare Guidance
        st.write("")
        st.markdown(f"""
        <div class="section-card">
            <h4 style="margin-top:0; color: #818CF8;">🩺 Preliminary Support & Guidance</h4>
            <p style="font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;">
                {get_guidance(disease)}
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer & Admin Panel Section
st.write("---")
admin_expander = st.expander(" Developer / Admin Controls", expanded=False)

with admin_expander:
    st.write("### Model Training & Analysis")
    
    col_admin1, col_admin2 = st.columns([1, 1])
    
    with col_admin1:
        if st.button("Train / Retrain Random Forest Model", use_container_width=True):
            if not DATA_PATH.exists():
                st.error(f"Missing training dataset: {DATA_PATH}")
            else:
                with st.spinner("Retraining model..."):
                    train_and_save_model()
                st.success("Model successfully retrained and saved locally!")
                st.rerun()
                
    with col_admin2:
        show_dataset = st.checkbox("Preview Training Dataset")
        
    if show_dataset:
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            st.write(f"Showing first 50 rows of `{DATA_PATH.name}`:")
            st.dataframe(df.head(50))
        else:
            st.error(f"Missing training dataset: {DATA_PATH}")
            
    st.write("")
