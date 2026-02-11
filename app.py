import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

# =====================================================
# Neural Network Core
# =====================================================
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

def train_network(X, Y, lr, epochs, hidden_neurons):
    np.random.seed(42)

    n_inputs = X.shape[1]
    n_outputs = 1

    W_hidden = np.random.randn(n_inputs, hidden_neurons) * 0.3
    B_hidden = np.zeros((1, hidden_neurons))
    W_output = np.random.randn(hidden_neurons, n_outputs) * 0.3
    B_output = np.zeros((1, n_outputs))

    loss_history = []
    activation_snapshots = []

    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        total_loss = 0
        indices = np.random.permutation(len(X))

        for i in indices:
            x = X[i:i+1]
            y = Y[i:i+1]

            # Forward
            Z_h = x @ W_hidden + B_hidden
            A_h = sigmoid(Z_h)
            Z_o = A_h @ W_output + B_output
            A_o = sigmoid(Z_o)

            error = A_o - y
            total_loss += error[0, 0] ** 2

            # Backprop (MSE + sigmoid)
            dA_o = error
            dZ_o = dA_o * sigmoid_derivative(A_o)

            dW_o = A_h.T @ dZ_o
            dB_o = dZ_o

            dA_h = dZ_o @ W_output.T
            dZ_h = dA_h * sigmoid_derivative(A_h)

            dW_h = x.T @ dZ_h
            dB_h = dZ_h

            # Update
            W_output -= lr * dW_o
            B_output -= lr * dB_o
            W_hidden -= lr * dW_h
            B_hidden -= lr * dB_h

        avg_loss = total_loss / len(X)
        loss_history.append(avg_loss)

        activation_snapshots.append({
            "hidden": A_h[0].tolist(),
            "output": float(A_o[0, 0])
        })

        progress.progress((epoch + 1) / epochs)
        status.text(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        time.sleep(0.005)

    return {
        "W_hidden": W_hidden,
        "B_hidden": B_hidden,
        "W_output": W_output,
        "B_output": B_output,
        "errors": loss_history,
        "activations": activation_snapshots
    }

def predict(x, model):
    h = sigmoid(x @ model["W_hidden"] + model["B_hidden"])
    o = sigmoid(h @ model["W_output"] + model["B_output"])
    return o

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config("🩺 Diabetes Neural Network", layout="wide")
st.title("🧠 Diabetes Neural Network Trainer")
st.caption("From Scratch • Training • Confidence • Interpretation")

# =====================================================
# Load Diabetes Dataset
# =====================================================
@st.cache_data
def load_diabetes():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age","Outcome"]
    df = pd.read_csv(url, names=cols)
    return df

df = load_diabetes()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# Select features
st.subheader("🧬 Feature Selection")
features = st.multiselect(
    "Choose input features",
    df.columns[:-1],
    default=["Glucose", "BMI", "Age"]
)

if len(features) == 0:
    st.warning("Select at least one feature.")
    st.stop()

X_raw = df[features].values.astype(np.float32)
Y = df["Outcome"].values.astype(np.float32).reshape(-1, 1)

# Scaling (VERY IMPORTANT)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# =====================================================
# Sidebar Controls
# =====================================================
st.sidebar.header("⚙️ Training Controls")
lr = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.03)
epochs = st.sidebar.slider("Epochs", 100, 3000, 1200)
hidden_neurons = st.sidebar.slider("Hidden Neurons", 2, 32, 8)

# =====================================================
# Training
# =====================================================
if st.button("🚀 Train Model", use_container_width=True):
    with st.spinner("Training neural network..."):
        model = train_network(X, Y, lr, epochs, hidden_neurons)
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["features"] = features

    st.success("✅ Training Completed")

    st.subheader("📉 Training Loss Curve")
    st.line_chart(model["errors"])

# =====================================================
# Model Evaluation
# =====================================================
if "model" in st.session_state:
    model = st.session_state["model"]

    preds = predict(X, model).flatten()
    actual = Y.flatten()

    st.subheader("📊 Actual vs Predicted")
    chart_df = pd.DataFrame({
        "Actual": actual,
        "Predicted": preds
    })
    st.scatter_chart(chart_df)

# =====================================================
# Manual Prediction
# =====================================================
if "model" in st.session_state:
    st.markdown("---")
    st.subheader("🔮 Predict Diabetes Risk (Manual Input)")

    user_inputs = []
    for f in st.session_state["features"]:
        val = st.number_input(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))
        user_inputs.append(val)

    if st.button("🧠 Predict", use_container_width=True):
        x_user = np.array(user_inputs).reshape(1, -1)
        x_user_scaled = st.session_state["scaler"].transform(x_user)

        prob = float(predict(x_user_scaled, st.session_state["model"])[0, 0])
        confidence = abs(prob - 0.5) * 2 * 100

        st.subheader("🎯 Prediction Result")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probability", f"{prob:.4f}")
        col2.metric("Risk Level", "High" if prob >= 0.5 else "Low")
        col3.metric("Confidence", f"{confidence:.1f}%")

        if prob >= 0.5:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

        with st.expander("ℹ️ How to interpret this"):
            st.write("""
            - Probability close to **1.0** → Strong diabetic signal  
            - Probability close to **0.0** → Strong non-diabetic signal  
            - Confidence measures distance from uncertainty (0.5)
            """)

