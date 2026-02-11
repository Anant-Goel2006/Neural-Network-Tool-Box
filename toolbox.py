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
st.title("🧠 Neural Network Learning Lab (Step-by-Step)")

# ======================================================
# Sidebar Controls
# ======================================================
menu = st.sidebar.selectbox(
    "📚 Learning Mode",
    ["Forward Propagation", "Backpropagation", "Full Gradient Descent Training"]
)

st.sidebar.markdown("---")
st.sidebar.header("📂 Dataset")

n_samples = st.sidebar.slider("Number of Samples", 1, 5, 2)

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

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Hyperparameters")

lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 1, 300, 50)

st.sidebar.markdown("---")
st.sidebar.header("🧮 Weight Initialization")

init_mode = st.sidebar.radio("Mode", ["Random", "Manual"])

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

# ======================================================
# Forward Pass Function
# ======================================================
def forward(x1, x2, w1, w2, w3, w4, w5, w6, b1, b2, b3):
    z1 = x1*w1 + x2*w2 + b1
    a1 = sigmoid(z1)

    z2 = x1*w3 + x2*w4 + b2
    a2 = sigmoid(z2)

    z3 = a1*w5 + a2*w6 + b3
    y_hat = sigmoid(z3)

    return z1, a1, z2, a2, z3, y_hat

# ======================================================
# FORWARD PROPAGATION MODE
# ======================================================
if menu == "Forward Propagation":

    st.header("🔵 Forward Propagation (Step-by-Step)")

    for i in range(len(X)):
        x1, x2 = X[i]
        z1, a1, z2, a2, z3, y_hat = forward(
            x1, x2, w1, w2, w3, w4, w5, w6, b1, b2, b3
        )

        st.subheader(f"Sample {i+1}")
        st.write(f"z1 = {x1}*{w1} + {x2}*{w2} + {b1} = {z1:.4f}")
        st.write(f"a1 = sigmoid(z1) = {a1:.4f}")
        st.write(f"z2 = {x1}*{w3} + {x2}*{w4} + {b2} = {z2:.4f}")
        st.write(f"a2 = sigmoid(z2) = {a2:.4f}")
        st.write(f"z3 = {a1:.4f}*{w5} + {a2:.4f}*{w6} + {b3} = {z3:.4f}")
        st.metric("Prediction (ŷ)", f"{y_hat:.4f}")

# ======================================================
# BACKPROPAGATION MODE
# ======================================================
elif menu == "Backpropagation":

    st.header("🟠 Backpropagation (Step-by-Step)")

    x1, x2 = X[0]
    y_true = Y[0][0]

    z1, a1, z2, a2, z3, y_hat = forward(
        x1, x2, w1, w2, w3, w4, w5, w6, b1, b2, b3
    )

    error = y_hat - y_true
    delta_output = error * sigmoid_derivative(y_hat)

    delta_h1 = delta_output * w5 * sigmoid_derivative(a1)
    delta_h2 = delta_output * w6 * sigmoid_derivative(a2)

    st.write(f"Error = ŷ - y = {y_hat:.4f} - {y_true} = {error:.4f}")
    st.write(f"δ_output = {delta_output:.6f}")
    st.write(f"δ_hidden1 = {delta_h1:.6f}")
    st.write(f"δ_hidden2 = {delta_h2:.6f}")

# ======================================================
# FULL TRAINING MODE
# ======================================================
elif menu == "Full Gradient Descent Training":

    st.header("🟢 Full Gradient Descent Training")

    # Local copies of weights
    w1_g, w2_g, w3_g, w4_g = w1, w2, w3, w4
    w5_g, w6_g = w5, w6
    b1_g, b2_g, b3_g = b1, b2, b3

    loss_history = []
    weight_history = []

    progress = st.progress(0)

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(X)):
            x1, x2 = X[i]
            y_true = Y[i][0]

            z1, a1, z2, a2, z3, y_hat = forward(
                x1, x2, w1_g, w2_g, w3_g, w4_g,
                w5_g, w6_g, b1_g, b2_g, b3_g
            )

            loss = 0.5 * (y_true - y_hat) ** 2
            total_loss += loss

            error = y_hat - y_true
            delta_output = error * sigmoid_derivative(y_hat)

            delta_h1 = delta_output * w5_g * sigmoid_derivative(a1)
            delta_h2 = delta_output * w6_g * sigmoid_derivative(a2)

            # Updates
            w5_g -= lr * delta_output * a1
            w6_g -= lr * delta_output * a2
            b3_g -= lr * delta_output

            w1_g -= lr * delta_h1 * x1
            w2_g -= lr * delta_h1 * x2
            b1_g -= lr * delta_h1

            w3_g -= lr * delta_h2 * x1
            w4_g -= lr * delta_h2 * x2
            b2_g -= lr * delta_h2

        loss_history.append(total_loss / len(X))
        weight_history.append(w1_g)
        progress.progress((epoch+1)/epochs)

    st.subheader("📉 Loss Curve")
    st.line_chart(loss_history)

    st.subheader("📈 Weight Evolution (w1)")
    st.line_chart(weight_history)

    st.success("Training Completed")
