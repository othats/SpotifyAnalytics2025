import streamlit as st
import shap
import joblib
import matplotlib.pyplot as plt
from utils import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Popularity Prediction & Explainability")

logo = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"
st.logo(logo)

df_audiofeatures, df_tracks = load_data()

# seleccionar columnas
numeric_cols = ['bpm','nrgy','dnce','dB','live','val','dur','acous','spch',]
categorical_cols = ['top genre']
id_cols = ['title', 'artist']
# asegurar que las columnas numéricas son del tipo correcto
for col in numeric_cols:
    df_audiofeatures[col] = pd.to_numeric(df_audiofeatures[col], errors='coerce')
df_audiofeatures = df_audiofeatures.dropna(subset=numeric_cols)
target = "pop"

# categorical features to string
for col in categorical_cols:
    df_audiofeatures[col] = df_audiofeatures[col].astype(str)

# remove nans
df_audiofeatures = df_audiofeatures.dropna(subset=numeric_cols + categorical_cols + [target])

# Load model
pipeline = joblib.load("models/popularity_pipeline.pkl")
rf_model = pipeline.named_steps["rf"]
preprocessor = pipeline.named_steps["pre"]

# Rebuild feature names
num_names = numeric_cols
cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
feature_names = ['bpm','nrgy','dnce','dB','live','val','dur','acous','spch'] + list(cat_names)

st.subheader("Model Performance on Test Set")

# Recalculate performance quickly
X = df_audiofeatures[numeric_cols + categorical_cols]
y = df_audiofeatures[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{rmse:.3f}")
with col2:
    st.metric("R² Score", f"{r2:.3f}")

st.divider()

# ===========================================================
# SECTION 1: Predict Popularity for a Custom Song
# ===========================================================

st.header("Predict Popularity for a New Track")

with st.form("pop_form"):
    cols = st.columns(3)
    user_inputs = {}

    # Numeric inputs
    for i, col in enumerate(numeric_cols):
        with cols[i % 3]:
            user_inputs[col] = st.number_input(
                col.replace("_", " ").title(),
                value=float(df_audiofeatures[col].mean())
            )

    # Categorical inputs
    for i, col in enumerate(categorical_cols):
        with cols[(i + len(numeric_cols)) % 3]:
            user_inputs[col] = st.selectbox(
                col.title(),
                sorted(df_audiofeatures[col].astype(str).unique())
            )

    submitted = st.form_submit_button("Predict Popularity")

if submitted:
    user_df = pd.DataFrame([user_inputs])
    pred_pop = pipeline.predict(user_df)[0]

    st.success(f"**Predicted Popularity: {pred_pop:.1f} / 100**")

    # SHAP for local explanation
    st.subheader("Explanation for This Prediction")

    explainer = shap.TreeExplainer(rf_model)

    # Transform user input into model-ready format
    transformed = preprocessor.transform(user_df)
    transformed_df = pd.DataFrame(
        transformed.toarray() if hasattr(transformed, "toarray") else transformed,
        columns=feature_names
    )

    shap_values_single = explainer(transformed_df)

    fig_local, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values_single[0], show=False)
    st.pyplot(fig_local)

st.divider()



# ===========================================================
# SECTION 2: Global Explainability (SHAP)
# ===========================================================

st.header("Global Explainability")

st.write("These plots show which features the model relies on most when predicting popularity across **all songs**.")

# Prepare full SHAP values only when user clicks (expensive)
if st.button("Compute Global SHAP Explanations"):
    with st.spinner("Computing SHAP values... this may take ~10-20 seconds"):
        X_train_trans = preprocessor.transform(X_train)
        X_train_df = pd.DataFrame(
            X_train_trans.toarray() if hasattr(X_train_trans, "toarray") else X_train_trans,
            columns=feature_names
        )

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer(X_train_df)

    # Beeswarm plot
    st.subheader("SHAP Beeswarm Plot")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_train_df, show=False)
    st.pyplot(fig1)

    # Bar plot
    st.subheader("Mean(|SHAP|) Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig2)

    st.info("""
### Interpretation

**Loudness (dB)** is the strongest predictor of popularity:  
- Louder tracks → higher predicted popularity  
- Softer recordings → lower predicted popularity  

**Genres** (via one-hot encoding) also have large influence.  
Some genres consistently add positive influence, others negative.

**Danceability, energy, valence, tempo** influence popularity in intuitive ways:
- High danceability boosts predictions  
- Very long / very short tracks reduce popularity  

**Acousticness & speechiness** often push predictions downward, meaning mainstream hits tend to be less acoustic and less speech-driven.

Overall, the model learns musically meaningful relationships.
""")
