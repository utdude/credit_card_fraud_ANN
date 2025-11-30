import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---- Page config ----
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ---- Custom light styling ----
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc; /* light slate */
    }
    h1, h2, h3, h4 {
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .section-box {
        padding:5px 5px;
        margin-bottom: 1rem;
        border-radius: 0.75rem;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb, #22c55e);
        color: white;
        border-radius: 999px;
        padding: 0.6rem 2.8rem;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.02em;
        cursor: pointer;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8, #16a34a);
        box-shadow: 0 4px 12px rgba(15,23,42,0.25);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Title ----
st.title("💳 Credit Card Fraud Detection")
st.caption("Enter recent transaction details. The app will compute the derived features used for prediction.")

# ================== SECTION 1: AMOUNTS ==================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("🧾 Transaction Amounts")

c1, c2, c3 = st.columns(3)

with c1:
    amount_t1 = st.number_input("Last Txn 1 (₹)", min_value=0.0, step=1.0, value=0.0)
with c2:
    amount_t2 = st.number_input("Last Txn 2 (₹)", min_value=0.0, step=1.0, value=0.0)
with c3:
    amount_t3 = st.number_input("Last Txn 3 (₹)", min_value=0.0, step=1.0, value=0.0)

current_amount = st.number_input("The Transaction Happened (₹)", min_value=0.0, step=1.0, value=0.0)

st.markdown('</div>', unsafe_allow_html=True)

# ================== SECTION 2: DISTANCES ==================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("📍 Distances")

d1, d2 = st.columns(2)
with d1:
    distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, step=0.5, value=0.0)
with d2:
    distance_from_last_transaction = st.number_input("Distance from Last Txn (km)", min_value=0.0, step=0.5, value=0.0)

st.markdown('</div>', unsafe_allow_html=True)

# ================== SECTION 3: BEHAVIOR FLAGS ==================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("⚙️ Transaction Behaviour")

b1, b2, b3, b4 = st.columns(4)
with b1:
    repeat_retailer = st.selectbox("Repeat Retailer?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with b2:
    used_chip = st.selectbox("Used Chip?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with b3:
    used_pin = st.selectbox("Used PIN?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with b4:
    online_order = st.selectbox("Online Order?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown('</div>', unsafe_allow_html=True)

# ================== ACTION BUTTON ==================
st.write("")  # small spacer
center_col = st.columns(3)[1]  # center the button

with center_col:
    show = st.button("Show Model Inputs")

# ================== PROCESS & DISPLAY ==================
if show:
    last_transactions = [amount_t1, amount_t2, amount_t3]
    # avoid divide-by-zero – if all are 0, set median to 1 (neutral)
    if all(v == 0 for v in last_transactions):
        median_past_amount = 1.0
    else:
        median_past_amount = float(np.median(last_transactions))

    if median_past_amount == 0:
        ratio_to_median_purchase_price = 0.0
    else:
        ratio_to_median_purchase_price = current_amount / median_past_amount

    ratio_to_median_purchase_price = round(ratio_to_median_purchase_price, 3)

    st.markdown("### 📊 Derived Feature")
    st.write(f"**Median of last 3 transactions:** ₹{median_past_amount:,.2f}")
    st.write(f"**Ratio to median purchase price:** `{ratio_to_median_purchase_price}`")

    # Final input row for the model
    input_data = pd.DataFrame({
        "distance_from_home": [distance_from_home],
        "distance_from_last_transaction": [distance_from_last_transaction],
        "ratio_to_median_purchase_price": [ratio_to_median_purchase_price],
        "repeat_retailer": [1 if repeat_retailer == 1 else 0],
        "used_chip": [1 if used_chip == 1 else 0],
        "used_pin_number": [1 if used_pin == 1 else 0],
        "online_order": [1 if online_order == 1 else 0],
    })

    model = load_model('creditcardfraud.h5')
    prediction = model.predict(input_data)

    if prediction[0][0] >= 0.5:
        st.error(f"**Model Prediction (Fraud Probability):** {prediction[0][0]:.4f} – Likely Fraudulent Transaction")
    else:
        st.success(f"**Model Prediction (Fraud Probability):** {prediction[0][0]:.4f} – Likely Legitimate Transaction")



