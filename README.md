# Segmentación de clientes con K-Means

Taller final de Minería de Datos: segmentación del *Mall Customers Dataset* con K-Means, método del codo, coeficiente de silueta, interpretación de negocio y despliegue con Streamlit.

## Archivos
- `Taller_KMeans_Mall_Customers.ipynb` – cuaderno con las Partes 1 a 7 y las respuestas.
- `app.py` – aplicación de Streamlit (codo, clusters y clasificación de un nuevo cliente).
- `requirements.txt` – dependencias.

## Aplicación
Desplegada en Streamlit Community Cloud: https://segmentacion-clientes-kmeans.streamlit.app

## Ejecución local
1. `pip install -r requirements.txt`
2. `streamlit run app.py` (la app descarga el dataset con `kagglehub` y entrena el modelo al iniciar)
