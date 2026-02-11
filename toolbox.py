import streamlit as st
import numpy as np
import pandas as pd

# ======================================================
# Activation
# ======================================================
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

# ======================================================
# Page Setup
# ======================================================
st.set_page_config(page_title="Neural Network Learning Toolbox", layout="wide")

# ======================================================
# Sidebar Navigation
# ======================================================
menu = st.sidebar.selectbox(
    "📚 Select Module",
    [
        "🏠 Home",
        "🔵 Forward Propagation",
        "🟠 Backpropagation",
        "🟢 Gradient Descent Training"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("📂 Dataset Options")

data_mode = st.sidebar.radio(
    "Choose Dataset Mode",
    ["Manual Input", "Upload CSV"]
)

# ======================================================
# Dataset
# ======================================================
if data_mode == "Manual Input":

    n_samples = st.sidebar.slider("Samples", 1, 10, 3)
    n_features = st.sidebar.slider("Features", 1, 5, 2)

    X = []
    Y = []

    for i in range(n_samples):
        row = []
        for j in range(n_features):
            val = st.sidebar.number_input(
                f"x{i}_f{j}", value=0.5, key=f"x{i}{j}"
            )
            row.append(val)

        y = st.sidebar.number_input(f"y_{i}", value=1.0, key=f"y{i}")
        X.append(row)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Dataset Loaded")

        target_column = st.sidebar.selectbox("Target Column", df.columns)
        feature_columns = st.sidebar.multiselect(
            "Feature Columns",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )

        X = df[feature_columns].values
        Y = df[target_column].values
    else:
        if menu != "🏠 Home":
            st.warning("Upload dataset to continue.")
            st.stop()
        X = np.array([[0.5, 0.8]])
        Y = np.array([1])

Y = Y.flatten()
n_features = X.shape[1]

# ======================================================
# Weight Initialization
# ======================================================
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Initialization")

init_mode = st.sidebar.radio("Mode", ["Random", "Manual"])

if init_mode == "Random":
    np.random.seed(42)
    w = np.random.randn(n_features)
else:
    w = np.array([
        st.sidebar.slider(f"w{i}", -2.0, 2.0, 0.5)
        for i in range(n_features)
    ])

b = st.sidebar.slider("Bias", -2.0, 2.0, 0.0)
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 1, 300, 50)

# ======================================================
# Forward Function
# ======================================================
def forward(x, w, b):
    z = np.dot(x, w) + b
    return z, sigmoid(z)

# ======================================================
# 🏠 HOME PAGE
# ======================================================
if menu == "🏠 Home":

    st.title("🧠 Neural Network Learning Toolbox")

    st.markdown("""
    This toolbox is an **interactive neural network simulator**
    designed to help students understand how neural networks work internally.

    ---
    ### 🔵 Forward Propagation
    Passes inputs through weights and activation functions  
    → Produces predictions  

    ### 🟠 Backpropagation
    Computes error and gradients  
    → Determines how weights should change  

    ### 🟢 Gradient Descent Training
    Iteratively updates weights  
    → Minimizes loss over epochs  

    ---
    ### How To Use:
    1. Select a module from sidebar  
    2. Upload or enter dataset  
    3. Initialize weights  
    4. Observe learning visually  
    """)

# ======================================================
# 🔵 FORWARD PROPAGATION
# ======================================================
elif menu == "🔵 Forward Propagation":

    st.header("Forward Propagation")

    predictions = []
    calculations = []

    for i in range(len(X)):
        z, y_hat = forward(X[i], w, b)
        predictions.append(y_hat)
        calculations.append((X[i], z, y_hat))

    st.metric("Average Prediction", f"{np.mean(predictions):.4f}")
    st.line_chart(pd.DataFrame({"Prediction": predictions}))

    with st.expander("🔍 See Detailed Calculations"):
        for i, (x, z, y_hat) in enumerate(calculations):
            st.write(f"Sample {i+1}")
            st.write(f"Input: {x}")
            st.write(f"Z = {z:.4f}")
            st.write(f"ŷ = {y_hat:.4f}")
            st.markdown("---")

# ======================================================
# 🟠 BACKPROPAGATION
# ======================================================
elif menu == "🟠 Backpropagation":

    st.header("Backpropagation")

    gradients = []
    details = []

    for i in range(len(X)):
        z, y_hat = forward(X[i], w, b)
        error = y_hat - Y[i]
        gradient = error * sigmoid_derivative(y_hat)

        gradients.append(gradient)
        details.append((y_hat, Y[i], error, gradient))

    st.metric("Average Gradient", f"{np.mean(gradients):.6f}")
    st.line_chart(pd.DataFrame({"Gradient": gradients}))

    with st.expander("🔍 See Detailed Calculations"):
        for i, (y_hat, y_true, error, grad) in enumerate(details):
            st.write(f"Sample {i+1}")
            st.write(f"Prediction: {y_hat:.4f}")
            st.write(f"True: {y_true}")
            st.write(f"Error: {error:.4f}")
            st.write(f"Gradient: {grad:.6f}")
            st.markdown("---")

# ======================================================
# 🟢 TRAINING
# ======================================================
elif menu == "🟢 Gradient Descent Training":

    st.header("Gradient Descent Training")

    w_train = w.copy()
    b_train = b

    loss_history = []
    weight_history = []

    for epoch in range(epochs):

        total_loss = 0

        for i in range(len(X)):
            z = np.dot(X[i], w_train) + b_train
            y_hat = sigmoid(z)

            error = y_hat - Y[i]
            loss = 0.5 * error**2
            total_loss += loss

            gradient = error * sigmoid_derivative(y_hat)

            w_train -= lr * gradient * X[i]
            b_train -= lr * gradient

        loss_history.append(total_loss / len(X))
        weight_history.append(np.mean(w_train))

    st.subheader("📉 Loss Curve")
    st.line_chart(pd.DataFrame({"Loss": loss_history}))

    st.subheader("📈 Weight Evolution")
    st.line_chart(pd.DataFrame({"Weight": weight_history}))

    with st.expander("🔍 See Training Details"):
        st.write("Final Weights:", w_train)
        st.write("Final Bias:", b_train)
        st.write("Final Loss:", loss_history[-1])

    st.success("Training Complete")
