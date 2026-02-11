import streamlit as st
import numpy as np
import pandas as pd

# ======================================================
# Activation Functions
# ======================================================
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

# ======================================================
# Page Setup
# ======================================================
st.set_page_config(page_title="Neural Network Learning Lab", layout="wide")
st.title("🧠 Neural Network Interactive Learning Lab")

# ======================================================
# Sidebar Menu
# ======================================================
menu = st.sidebar.selectbox(
    "📚 Select Module",
    ["Forward Propagation", "Backpropagation", "Gradient Descent Training"]
)

st.sidebar.markdown("---")
st.sidebar.header("📂 Dataset Options")

data_mode = st.sidebar.radio(
    "Choose Dataset Mode",
    ["Manual Input", "Upload CSV"]
)

# ======================================================
# Dataset Section
# ======================================================
if data_mode == "Manual Input":
    st.sidebar.markdown("### Enter Dataset (2 Features)")
    n_samples = st.sidebar.slider("Number of Samples", 1, 10, 3)

    X = []
    Y = []

    for i in range(n_samples):
        st.sidebar.markdown(f"Sample {i+1}")
        x1 = st.sidebar.number_input(f"x1_{i}", value=0.5, key=f"x1{i}")
        x2 = st.sidebar.number_input(f"x2_{i}", value=0.8, key=f"x2{i}")
        y = st.sidebar.number_input(f"y_{i}", value=1.0, key=f"y{i}")
        X.append([x1, x2])
        Y.append([y])

    X = np.array(X)
    Y = np.array(Y)

else:
    uploaded = st.sidebar.file_uploader("Upload CSV (2 features + 1 target)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        X = df.iloc[:, :-1].values
        Y = df.iloc[:, -1].values.reshape(-1, 1)
        st.write("### Uploaded Dataset")
        st.dataframe(df.head())
    else:
        st.warning("Upload a dataset to proceed.")
        st.stop()

# ======================================================
# Weight Initialization
# ======================================================
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Weight Initialization")

init_mode = st.sidebar.radio("Initialization Mode", ["Random", "Manual"])

if init_mode == "Random":
    np.random.seed(42)
    w1, w2, w3, w4 = np.random.randn(4)
    w5, w6 = np.random.randn(2)
    b1, b2, b3 = np.random.randn(3)
else:
    w1 = st.sidebar.slider("w1", -2.0, 2.0, 0.5)
    w2 = st.sidebar.slider("w2", -2.0, 2.0, -0.4)
    w3 = st.sidebar.slider("w3", -2.0, 2.0, 0.3)
    w4 = st.sidebar.slider("w4", -2.0, 2.0, 0.1)
    w5 = st.sidebar.slider("w5", -2.0, 2.0, 0.7)
    w6 = st.sidebar.slider("w6", -2.0, 2.0, -0.2)
    b1 = st.sidebar.slider("b1", -2.0, 2.0, 0.0)
    b2 = st.sidebar.slider("b2", -2.0, 2.0, 0.0)
    b3 = st.sidebar.slider("b3", -2.0, 2.0, 0.0)

lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 1, 500, 50)

# ======================================================
# Forward Function
# ======================================================
def forward(x1, x2):
    z1 = x1*w1 + x2*w2 + b1
    a1 = sigmoid(z1)

    z2 = x1*w3 + x2*w4 + b2
    a2 = sigmoid(z2)

    z3 = a1*w5 + a2*w6 + b3
    y_hat = sigmoid(z3)

    return z1, a1, z2, a2, z3, y_hat

# ======================================================
# FORWARD PROPAGATION
# ======================================================
if menu == "Forward Propagation":
    st.header("🔵 Forward Propagation")

    for i in range(len(X)):
        x1, x2 = X[i]
        z1, a1, z2, a2, z3, y_hat = forward(x1, x2)

        st.subheader(f"Sample {i+1}")
        st.write(f"z1 = {z1:.4f}, a1 = {a1:.4f}")
        st.write(f"z2 = {z2:.4f}, a2 = {a2:.4f}")
        st.metric("Prediction (ŷ)", f"{y_hat:.4f}")

# ======================================================
# BACKPROPAGATION
# ======================================================
elif menu == "Backpropagation":
    st.header("🟠 Backpropagation")

    x1, x2 = X[0]
    y_true = Y[0][0]

    z1, a1, z2, a2, z3, y_hat = forward(x1, x2)

    error = y_hat - y_true
    delta_output = error * sigmoid_derivative(y_hat)

    delta_h1 = delta_output * w5 * sigmoid_derivative(a1)
    delta_h2 = delta_output * w6 * sigmoid_derivative(a2)

    st.write("Error:", error)
    st.write("Output Gradient:", delta_output)
    st.write("Hidden Gradient 1:", delta_h1)
    st.write("Hidden Gradient 2:", delta_h2)

# ======================================================
# GRADIENT DESCENT TRAINING
# ======================================================
elif menu == "Gradient Descent Training":
    st.header("🟢 Gradient Descent Training")

    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            x1, x2 = X[i]
            y_true = Y[i][0]

            z1, a1, z2, a2, z3, y_hat = forward(x1, x2)

            loss = 0.5 * (y_true - y_hat) ** 2
            total_loss += loss

            error = y_hat - y_true
            delta_output = error * sigmoid_derivative(y_hat)

            delta_h1 = delta_output * w5 * sigmoid_derivative(a1)
            delta_h2 = delta_output * w6 * sigmoid_derivative(a2)

            # Update weights
            w5 -= lr * delta_output * a1
            w6 -= lr * delta_output * a2
            b3 -= lr * delta_output

            w1 -= lr * delta_h1 * x1
            w2 -= lr * delta_h1 * x2
            b1 -= lr * delta_h1

            w3 -= lr * delta_h2 * x1
            w4 -= lr * delta_h2 * x2
            b2 -= lr * delta_h2

        loss_history.append(total_loss / len(X))

    st.subheader("📉 Loss Curve")
    st.line_chart(loss_history)

    st.success("Training Complete")
