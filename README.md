# Segmentación de clientes con K-Means

**Taller Final – Inteligencia de Negocios y Minería de Datos**  
Universidad Cooperativa de Colombia  
**Integrantes:** Jack Anderson Limas Solarte · Juan David Maya Benavides  
**Profesor:** Cristian Camilo Ordoñez Quintero  
**Fecha:** 22/10/2026

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanMaya8/segmentacion_clientes/blob/main/Taller_KMeans_Mall_Customers.ipynb)
[![Abrir la app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segmentacion-clientes-juanmaya8.streamlit.app)

## Descripción
En este proyecto segmentamos a los clientes del *Mall Customers Dataset* (Kaggle) con el algoritmo K-Means. Usamos el ingreso anual y el Spending Score, elegimos el número de clusters con el método del codo y el coeficiente de silueta, interpretamos cada grupo desde el punto de vista del negocio y desplegamos una aplicación en Streamlit para clasificar clientes nuevos.

## Cómo revisar el trabajo
1. **Cuaderno:** con el botón **Open in Colab** se abre `Taller_KMeans_Mall_Customers.ipynb` ya ejecutado, con todas las gráficas, tablas y respuestas. También se puede ver directamente en GitHub. Si se quiere volver a ejecutar: *Entorno de ejecución → Ejecutar todas* (el dataset se descarga solo con `kagglehub`).
2. **Aplicación:** https://segmentacion-clientes-juanmaya8.streamlit.app  
   Si lleva unos días sin visitas, la app entra en reposo; al abrirla aparece un botón para activarla y tarda menos de un minuto.

## Contenido del cuaderno
| Parte | Contenido |
|---|---|
| 1. Exploración de datos | Carga del dataset, variables relevantes, estadística descriptiva, atípicos y relaciones entre variables. |
| 2. Preprocesamiento | Normalización con `StandardScaler`. |
| 3. Método del codo | Inercia para K = 1 a 10. |
| 4. Coeficiente de silueta | Silueta para K = 2 a 10 y gráfico de silueta por observación. |
| 5. Modelo K-Means | Entrenamiento con K = 5, asignación de clusters y centroides. |
| 6. Interpretación de negocio | Perfiles de clientes y estrategias de marketing. |
| 7. Despliegue con Streamlit | Enlace a la aplicación y explicación de lo que contiene. |

## Resultados principales
- El método del codo y el coeficiente de silueta (0.555) coinciden en **K = 5**.
- Perfiles obtenidos: clientes de alto valor, promedio, con potencial, de bajo valor y de alto gasto con bajo ingreso.
- El segmento con mayor oportunidad de crecimiento es el de **ingreso alto y gasto bajo**.

## Archivos
- `Taller_KMeans_Mall_Customers.ipynb` – cuaderno con las Partes 1 a 7, resultados y respuestas.
- `app.py` – aplicación de Streamlit (método del codo, clusters y clasificación de un nuevo cliente).
- `requirements.txt` – dependencias de la aplicación.

## Ejecutar la aplicación localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
