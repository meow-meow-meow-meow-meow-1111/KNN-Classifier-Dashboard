import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Page configuration
st.set_page_config(page_title="IQROGUEREX - K-NN Classifier Dashboard", layout="wide")

st.title("🛡️ K-Nearest Neighbors (K-NN) Classifier")
st.markdown(f"**Developed for {st.session_state.get('user_name', 'Chinmay V Chatradamath')} @ IQROGUEREX**")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('Social_Network_Ads.csv')
    return df

df = load_data()

# --- Sidebar - Model Parameters ---
st.sidebar.header("Model Parameters")
n_neighbors = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 25) / 100

# --- Data Preparation ---
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# --- Model Training ---
classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='minkowski', p=2)
classifier.fit(X_train_scaled, y_train)

# --- Layout Columns ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Predict New Result")
    input_age = st.number_input("Age", min_value=18, max_value=60, value=30)
    input_salary = st.number_input("Estimated Salary", min_value=15000, max_value=150000, value=87000)
    
    if st.button("Predict"):
        prediction = classifier.predict(sc.transform([[input_age, input_salary]]))
        result = "Purchased" if prediction[0] == 1 else "Not Purchased"
        st.success(f"Prediction: **{result}**")

    # Metrics
    y_pred = classifier.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, 
                       labels=dict(x="Predicted", y="Actual"),
                       x=['Not Purchased', 'Purchased'],
                       y=['Not Purchased', 'Purchased'],
                       color_continuous_scale='Blues',
                       title="Confusion Matrix")
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.subheader("Decision Boundary Visualization")
    
    # Create meshgrid for Plotly contour
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
    
    # Create steps for mesh
    x_range = np.linspace(x_min, x_max, 100)
    y_range = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Predict over mesh
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_preds = classifier.predict(sc.transform(grid_points))
    grid_preds = grid_preds.reshape(xx.shape)

    # Plotly Figure
    fig = go.Figure()

    # Add Decision Boundary (Contour)
    fig.add_trace(go.Contour(
        x=x_range, y=y_range, z=grid_preds,
        showscale=False,
        colorscale=[[0, '#FA8072'], [1, '#1E90FF']],
        opacity=0.4,
        hoverinfo='skip'
    ))

    # Add Scatter points for Test Set
    for i, label in enumerate(['Not Purchased', 'Purchased']):
        mask = y_test == i
        fig.add_trace(go.Scatter(
            x=X_test[mask, 0],
            y=X_test[mask, 1],
            mode='markers',
            name=label,
            marker=dict(color='#FA8072' if i==0 else '#1E90FF', size=10, line=dict(width=1, color='Black'))
        ))

    fig.update_layout(
        title=f"K-NN Decision Boundary (K={n_neighbors})",
        xaxis_title="Age",
        yaxis_title="Estimated Salary",
        legend_title="Outcome",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Raw Data Display
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df)
