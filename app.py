import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de clientes", layout="wide")
st.title("🛍️ Segmentación de clientes con K-Means")
st.caption("Mall Customers Dataset – Ingreso anual vs Spending Score")

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_kmeans.pkl")

@st.cache_data
def cargar_datos():
    return pd.read_csv("Mall_Customers_clusters.csv")

modelo = cargar_modelo()
df = cargar_datos()
scaler, kmeans, perfiles, features = modelo["scaler"], modelo["kmeans"], modelo["perfiles"], modelo["features"]
X_scaled = scaler.transform(df[features].values)
centroides = scaler.inverse_transform(kmeans.cluster_centers_)
colores = plt.cm.tab10.colors

@st.cache_data
def curvas(_X):
    ks = list(range(1, 11)); inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(_X)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(_X, km.labels_) if k > 1 else np.nan)
    return ks, inercias, siluetas

tab1, tab2, tab3 = st.tabs(["📉 Método del codo", "🎯 Clusters", "🧑 Clasificar nuevo cliente"])

with tab1:
    ks, inercias, siluetas = curvas(X_scaled)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.plot(ks, inercias, "o-"); ax.axvline(kmeans.n_clusters, color="red", ls="--")
        ax.set_xlabel("K"); ax.set_ylabel("Inercia (WCSS)"); ax.set_title("Método del codo")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.plot(ks[1:], siluetas[1:], "o-", color="green"); ax.axvline(kmeans.n_clusters, color="red", ls="--")
        ax.set_xlabel("K"); ax.set_ylabel("Silueta"); ax.set_title("Coeficiente de silueta")
        st.pyplot(fig)
    st.info(f"K seleccionado: **{kmeans.n_clusters}** (codo y máxima silueta)")

with tab2:
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(df["Cluster"].unique()):
        d = df[df["Cluster"] == c]
        ax.scatter(d["Income"], d["Score"], s=50, color=colores[c], label=f"{c}: {perfiles[c]}")
    ax.scatter(centroides[:, 0], centroides[:, 1], c="black", s=250, marker="X", label="Centroides")
    ax.set_xlabel("Ingreso anual (k$)"); ax.set_ylabel("Spending Score (1-100)")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    resumen = df.groupby("Cluster").agg(Clientes=("CustomerID", "count"), Edad_media=("Age", "mean"),
                                        Ingreso_medio=("Income", "mean"), Score_medio=("Score", "mean")).round(1)
    resumen["Perfil"] = resumen.index.map(perfiles)
    st.dataframe(resumen, use_container_width=True)

with tab3:
    st.subheader("Datos del nuevo cliente")
    c1, c2 = st.columns(2)
    ingreso = c1.number_input("Ingreso anual (k$)", min_value=0, max_value=300, value=60, step=1)
    score = c2.slider("Spending Score (1-100)", 1, 100, 50)
    if st.button("Clasificar cliente", type="primary"):
        nuevo = scaler.transform(np.array([[ingreso, score]]))
        cl = int(kmeans.predict(nuevo)[0])
        st.success(f"El cliente pertenece al **cluster {cl}** → **{perfiles[cl]}**")
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in sorted(df["Cluster"].unique()):
            d = df[df["Cluster"] == c]
            ax.scatter(d["Income"], d["Score"], s=40, alpha=0.5, color=colores[c], label=f"{c}: {perfiles[c]}")
        ax.scatter(ingreso, score, c="red", s=400, marker="*", edgecolor="black", label="Nuevo cliente")
        ax.set_xlabel("Ingreso anual (k$)"); ax.set_ylabel("Spending Score (1-100)"); ax.legend(fontsize=8)
        st.pyplot(fig)