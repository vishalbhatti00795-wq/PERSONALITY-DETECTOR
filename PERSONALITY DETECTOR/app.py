import streamlit as st
import pickle
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Personality Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #111827 50%,
            #1e1b4b 100%
        );
        color: white;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.3rem;
        background: linear-gradient(
            90deg,
            #a78bfa,
            #60a5fa,
            #c084fc
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e9d5ff;
        margin-bottom: 15px;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.35),
            rgba(37, 99, 235, 0.25)
        );
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 22px;
        padding: 35px;
        text-align: center;
        margin-top: 25px;
    }

    .result-title {
        font-size: 1.1rem;
        color: #cbd5e1;
    }

    .result-personality {
        font-size: 2.7rem;
        font-weight: 800;
        color: white;
        margin: 10px 0;
    }

    .confidence {
        font-size: 1.2rem;
        color: #ddd6fe;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1rem;
        font-size: 1rem;
        font-weight: 700;
        background: linear-gradient(
            90deg,
            #7c3aed,
            #2563eb
        );
        color: white;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.35);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    with open("personality_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


try:
    model, scaler = load_models()

except FileNotFoundError:
    st.error(
        "Model files not found. Make sure personality_model.pkl "
        "and scaler.pkl are in the same folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 Personality Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover whether your personality leans toward '
    '<b>Introvert</b>, <b>Ambivert</b>, or <b>Extrovert</b>'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## 🧠 Personality Detector")

    st.markdown("""
    ### How it works

    Rate yourself from **0 to 10** based on your typical behavior.

    The Logistic Regression model will analyze your responses and
    predict your personality type.

    **Possible results:**

    - 🟣 Introvert
    - 🔵 Ambivert
    - 🟢 Extrovert
    """)

    st.markdown("---")

    st.caption("Machine Learning Project")
    st.caption("Algorithm: Logistic Regression")


# ---------------------------------------------------------
# FEATURE INPUTS
# ---------------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🌐 Social & Communication</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    social_energy = st.slider(
        "Social Energy",
        0.0, 10.0, 5.0,
        help="How energetic do you feel around other people?"
    )

    talkativeness = st.slider(
        "Talkativeness",
        0.0, 10.0, 5.0,
        help="How much do you usually enjoy talking?"
    )

    group_comfort = st.slider(
        "Group Comfort",
        0.0, 10.0, 5.0,
        help="How comfortable are you in groups?"
    )

    party_liking = st.slider(
        "Party Liking",
        0.0, 10.0, 5.0,
        help="How much do you enjoy parties and social events?"
    )

    listening_skill = st.slider(
        "Listening Skill",
        0.0, 10.0, 7.0
    )

    empathy = st.slider(
        "Empathy",
        0.0, 10.0, 6.0
    )

with col2:

    alone_time_preference = st.slider(
        "Alone Time Preference",
        0.0, 10.0, 5.0,
        help="How much do you enjoy spending time alone?"
    )

    public_speaking_comfort = st.slider(
        "Public Speaking Comfort",
        0.0, 10.0, 5.0
    )

    friendliness = st.slider(
        "Friendliness",
        0.0, 10.0, 6.0
    )

    online_social_usage = st.slider(
        "Online Social Usage",
        0.0, 10.0, 6.0
    )

    work_style_collaborative = st.slider(
        "Collaborative Work Style",
        0.0, 10.0, 6.0
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# THINKING & PERSONALITY
# ---------------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🧩 Thinking & Personality</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    deep_reflection = st.slider(
        "Deep Reflection",
        0.0, 10.0, 6.0
    )

    organization = st.slider(
        "Organization",
        0.0, 10.0, 6.0
    )

    leadership = st.slider(
        "Leadership",
        0.0, 10.0, 6.0
    )

    curiosity = st.slider(
        "Curiosity",
        0.0, 10.0, 6.0
    )

    routine_preference = st.slider(
        "Routine Preference",
        0.0, 10.0, 5.0
    )

with col2:

    risk_taking = st.slider(
        "Risk Taking",
        0.0, 10.0, 5.0
    )

    excitement_seeking = st.slider(
        "Excitement Seeking",
        0.0, 10.0, 5.0
    )

    planning = st.slider(
        "Planning",
        0.0, 10.0, 6.0
    )

    spontaneity = st.slider(
        "Spontaneity",
        0.0, 10.0, 5.0
    )

    adventurousness = st.slider(
        "Adventurousness",
        0.0, 10.0, 5.0
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# INTERESTS & LIFESTYLE
# ---------------------------------------------------------

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🎯 Interests & Lifestyle</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    creativity = st.slider(
        "Creativity",
        0.0, 10.0, 6.0
    )

    reading_habit = st.slider(
        "Reading Habit",
        0.0, 10.0, 6.0
    )

    sports_interest = st.slider(
        "Sports Interest",
        0.0, 10.0, 5.0
    )

with col2:

    travel_desire = st.slider(
        "Travel Desire",
        0.0, 10.0, 6.0
    )

    gadget_usage = st.slider(
        "Gadget Usage",
        0.0, 10.0, 6.0
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.markdown("### 🔮 Ready to discover your personality?")


if st.button("✨ Detect My Personality"):

    # -----------------------------------------------------
    # IMPORTANT:
    # Your model was trained WITHOUT:
    # creativity
    #
    # Therefore creativity is intentionally NOT passed
    # to the model.
    # -----------------------------------------------------

    input_data = np.array([[
        social_energy,
        alone_time_preference,
        talkativeness,
        deep_reflection,
        group_comfort,
        party_liking,
        listening_skill,
        empathy,

        organization,
        leadership,
        risk_taking,
        public_speaking_comfort,
        curiosity,
        routine_preference,
        excitement_seeking,
        friendliness,

        planning,
        spontaneity,
        adventurousness,
        reading_habit,
        sports_interest,
        online_social_usage,
        travel_desire,
        gadget_usage,
        work_style_collaborative,
        decision_speed if 'decision_speed' in locals() else 5.0
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probabilities = model.predict_proba(input_scaled)[0]

    # LabelEncoder mapping from your notebook:
    # Ambivert = 0
    # Extrovert = 1
    # Introvert = 2

    personality_map = {
        0: "Ambivert",
        1: "Extrovert",
        2: "Introvert"
    }

    personality = personality_map.get(
        int(prediction),
        str(prediction)
    )

    confidence = float(np.max(probabilities)) * 100


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-title">
                Your predicted personality type is
            </div>

            <div class="result-personality">
                {personality}
            </div>

            <div class="confidence">
                Model confidence: <b>{confidence:.2f}%</b>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # PROBABILITY BREAKDOWN
    # -----------------------------------------------------

    st.markdown("### 📊 Prediction Breakdown")

    probability_df = pd.DataFrame({
        "Personality": [
            "Ambivert",
            "Extrovert",
            "Introvert"
        ],
        "Probability": probabilities * 100
    })

    cols = st.columns(3)

    for i, row in probability_df.iterrows():

        with cols[i]:

            st.metric(
                row["Personality"],
                f"{row['Probability']:.2f}%"
            )

            st.progress(
                min(int(row["Probability"]), 100)
            )


    # -----------------------------------------------------
    # PERSONALITY DESCRIPTION
    # -----------------------------------------------------

    descriptions = {

        "Introvert":
            "You may prefer quieter environments, "
            "meaningful conversations, and spending time "
            "recharging on your own.",

        "Ambivert":
            "You appear to have a balance between social "
            "interaction and personal space. Your behavior "
            "may adapt depending on the situation.",

        "Extrovert":
            "You may gain energy from social interaction, "
            "enjoy communicating with others, and feel "
            "comfortable in active social environments."
    }

    st.markdown("### 💡 What this means")

    st.info(descriptions.get(
        personality,
        "The model has made a prediction based on your inputs."
    ))


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;">
        Built with Python • Scikit-learn • Logistic Regression • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)