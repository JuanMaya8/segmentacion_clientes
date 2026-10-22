import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Segmentación de clientes", page_icon="🛍️", layout="wide")

FEATURES = ["Income", "Score"]
K_OPT = 5
COLORES = plt.cm.tab10.colors


# ---------------------------------------------------------------- datos
@st.cache_data
def cargar_datos():
    if os.path.exists("Mall_Customers.csv"):
        ruta = "Mall_Customers.csv"
    else:
        import kagglehub
        carpeta = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
        ruta = os.path.join(carpeta, "Mall_Customers.csv")
    df = pd.read_csv(ruta)
    return df.rename(columns={"Genre": "Gender",
                              "Annual Income (k$)": "Income",
                              "Spending Score (1-100)": "Score"})


def perfil(ing, sco, ing_med, sco_med):
    if abs(ing - ing_med) < 15 and abs(sco - sco_med) < 15:
        return "Clientes promedio"
    if ing >= ing_med and sco >= sco_med:
        return "Clientes de alto valor"
    if ing >= ing_med and sco < sco_med:
        return "Clientes con potencial"
    if ing < ing_med and sco >= sco_med:
        return "Alto gasto con bajo ingreso"
    return "Clientes de bajo valor"


# ---------------------------------------------------------------- modelo
@st.cache_resource
def entrenar(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)

    ks = list(range(1, 11))
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X, km.labels_) if k > 1 else np.nan)

    kmeans = KMeans(n_clusters=K_OPT, init="k-means++", n_init=10, random_state=42).fit(X)
    centroides = scaler.inverse_transform(kmeans.cluster_centers_)
    ing_med, sco_med = df["Income"].median(), df["Score"].median()
    perfiles = {i: perfil(c[0], c[1], ing_med, sco_med) for i, c in enumerate(centroides)}
    return scaler, kmeans, centroides, perfiles, ks, inercias, siluetas


df = cargar_datos().copy()
scaler, kmeans, centroides, perfiles, ks, inercias, siluetas = entrenar(df)
df["Cluster"] = kmeans.labels_
df["Perfil"] = df["Cluster"].map(perfiles)


def grafico_clusters(nuevo=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(K_OPT):
        d = df[df["Cluster"] == c]
        ax.scatter(d["Income"], d["Score"], s=45, alpha=0.75 if nuevo is None else 0.45,
                   color=COLORES[c], label=f"Cluster {c}: {perfiles[c]}")
    ax.scatter(centroides[:, 0], centroides[:, 1], c="black", s=250, marker="X", label="Centroides")
    if nuevo is not None:
        ax.scatter(*nuevo, c="red", s=450, marker="*", edgecolor="black", label="Nuevo cliente")
    ax.set_xlabel("Ingreso anual (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- interfaz
st.title("🛍️ Segmentación de clientes con K-Means")
st.caption("Mall Customers Dataset · variables: ingreso anual y Spending Score · K = 5")

c1, c2, c3 = st.columns(3)
c1.metric("Clientes", len(df))
c2.metric("Clusters (K)", K_OPT)
c3.metric("Coeficiente de silueta", f"{siluetas[K_OPT - 1]:.3f}")

tab1, tab2, tab3 = st.tabs(["📉 Método del codo", "🎯 Clusters", "🧑 Clasificar nuevo cliente"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(ks, inercias, "o-", linewidth=2)
        ax.axvline(K_OPT, color="red", ls="--", label=f"K = {K_OPT}")
        ax.set_xlabel("Número de clusters (K)"); ax.set_ylabel("Inercia (WCSS)")
        ax.set_title("Método del codo"); ax.set_xticks(ks); ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.plot(ks[1:], siluetas[1:], "o-", color="green", linewidth=2)
        ax.axvline(K_OPT, color="red", ls="--", label=f"K = {K_OPT}")
        ax.set_xlabel("Número de clusters (K)"); ax.set_ylabel("Silueta promedio")
        ax.set_title("Coeficiente de silueta"); ax.set_xticks(ks[1:]); ax.legend()
        st.pyplot(fig)
    tabla = pd.DataFrame({"K": ks, "Inercia": np.round(inercias, 2), "Silueta": np.round(siluetas, 4)})
    st.dataframe(tabla, hide_index=True, use_container_width=True)
    st.info("El codo de la curva y el máximo de la silueta coinciden en **K = 5**.")

with tab2:
    st.pyplot(grafico_clusters())
    resumen = df.groupby("Cluster").agg(Clientes=("CustomerID", "count"),
                                        Edad_media=("Age", "mean"),
                                        Ingreso_medio=("Income", "mean"),
                                        Score_medio=("Score", "mean")).round(1)
    resumen["Perfil"] = resumen.index.map(perfiles)
    st.dataframe(resumen, use_container_width=True)

with tab3:
    st.subheader("Datos del nuevo cliente")
    col1, col2 = st.columns(2)
    ingreso = col1.number_input("Ingreso anual (k$)", min_value=0, max_value=300, value=60, step=1)
    score = col2.slider("Spending Score (1-100)", 1, 100, 50)
    if st.button("Clasificar cliente", type="primary"):
        cl = int(kmeans.predict(scaler.transform([[ingreso, score]]))[0])
        st.success(f"El cliente pertenece al **cluster {cl}** → **{perfiles[cl]}**")
        st.pyplot(grafico_clusters((ingreso, score)))
