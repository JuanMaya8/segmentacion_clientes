# Segmentación de clientes con K-Means

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanMaya8/segmentacion_clientes/blob/main/Taller_KMeans_Mall_Customers.ipynb)
[![Abrir la app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segmentacion-clientes-juanmaya8.streamlit.app)

Taller final de Minería de Datos: segmentación del *Mall Customers Dataset* con K-Means, método del codo, coeficiente de silueta, interpretación de negocio y despliegue con Streamlit.

## Archivos
- `Taller_KMeans_Mall_Customers.ipynb` – cuaderno con las Partes 1 a 7, sus resultados y las respuestas.
- `app.py` – aplicación de Streamlit (método del codo, clusters y clasificación de un nuevo cliente).
- `requirements.txt` – dependencias.

## Resultados principales
- Variables usadas: ingreso anual y Spending Score, escaladas con `StandardScaler`.
- El método del codo y el coeficiente de silueta (0,555) coinciden en **K = 5**.
- Perfiles: clientes de alto valor, promedio, con potencial, de bajo valor y de alto gasto con bajo ingreso.

## Aplicación en Streamlit
Desplegada en Streamlit Community Cloud: https://segmentacion-clientes-juanmaya8.streamlit.app

Si lleva varios días sin visitas, la app entra en reposo; al abrirla aparece un botón para despertarla y tarda menos de un minuto.

## Abrir y editar el cuaderno en Google Colab
1. Pulsa el botón **Abrir en Colab** de arriba, o entra a:
   https://colab.research.google.com/github/JuanMaya8/segmentacion_clientes/blob/main/Taller_KMeans_Mall_Customers.ipynb
   (cualquier cuaderno de GitHub se abre en Colab cambiando `github.com` por `colab.research.google.com/github` en su dirección).
2. Para ejecutarlo: **Entorno de ejecución → Ejecutar todas**. El dataset se descarga solo con `kagglehub`.
3. Para guardar los cambios en este repositorio: **Archivo → Guardar una copia en GitHub**, con el repositorio `JuanMaya8/segmentacion_clientes`, la rama `main` y la ruta `Taller_KMeans_Mall_Customers.ipynb`.

> **Importante:** abre siempre el cuaderno desde GitHub justo antes de editarlo. Si guardas desde una pestaña de Colab abierta antes de un cambio hecho en el repositorio, ese cambio se sobrescribe. Si acabas de hacer un commit, espera unos minutos antes de abrirlo para que Colab no cargue una copia anterior.

## Ejecución local de la app
1. `pip install -r requirements.txt`
2. `streamlit run app.py` (la app descarga el dataset con `kagglehub` y entrena el modelo al iniciar)
