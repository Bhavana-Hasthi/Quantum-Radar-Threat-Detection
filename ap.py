import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

st.set_page_config(page_title="Quantum Radar Threat Detection", layout="wide")

# Navigation bar
page = st.sidebar.radio("Navigation", ["Overview", "Simulation", "Results & Visuals", "Conclusion"])

# Overview Page
if page == "Overview":
    st.title("🚀 Quantum-Enhanced Radar Threat Detection System")
    st.markdown("""
    This project demonstrates how quantum-inspired convolutional neural networks (CNNs) can enhance radar threat detection,
    especially for stealth aircraft or low-observable threats.

    ### Key Highlights:
    - Uses simulated radar signal data
    - Incorporates CNNs as a baseline (can be extended to Quantum CNNs)
    - Performance metrics: Accuracy, Precision, Recall, F1-Score
    - Easy-to-understand visuals and interaction
    """)

    st.image("radar image.png", caption="Live Radar Scan Visualization", use_container_width=True)

# Simulation Page
elif page == "Simulation":
    st.header("⚙️ Simulation Settings")
    num_samples = st.slider("Number of Samples", 100, 2000, 500, step=100)
    epochs = st.slider("Epochs", 2, 20, 5)
    batch_size = st.slider("Batch Size", 8, 64, 16)

    if st.button("Run Radar Simulation"):
        X = np.random.rand(num_samples, 100, 1)
        y = np.random.randint(0, 2, num_samples)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 1)),
            MaxPooling1D(2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.success("Simulation Complete! 📡")

# Results & Visuals Page
elif page == "Results & Visuals":
    st.header("📊 Results and Visualizations")

    if 'y_test' in st.session_state and 'y_pred' in st.session_state:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']

        report = classification_report(y_test, y_pred, output_dict=True)
        st.text("Performance Metrics")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        st.subheader("Prediction Distribution")
        fig2, ax2 = plt.subplots()
        bars = np.bincount(y_pred.flatten())
        ax2.bar(['Class 0', 'Class 1'], bars, color=['orange', 'cyan'])
        ax2.set_title("Distribution of Predicted Classes")
        st.pyplot(fig2)

    else:
        st.warning("Please run the simulation first.")

# Conclusion Page
elif page == "Conclusion":
    st.header("🛰️ Conclusion")
    st.markdown("""
    This demo showcased how machine learning, specifically CNNs (with potential for quantum models),
    can significantly enhance radar systems for modern-day defense applications.

    ### Summary:
    - Real-time simulation of radar signals
    - Classification between stealth and visible objects
    - Visualization of performance with accuracy and confusion matrix

    ### Future Improvements:
    - Integrate Pennylane for true Quantum CNNs
    - Use real radar datasets with preprocessing
    - Deploy on cloud for real-time applications

    Thank you for watching the demo! 💡
    """)
