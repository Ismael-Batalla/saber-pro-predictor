# Proyecto final – Sistemas Inteligentes: Predictor Saber Pro

Este repositorio contiene el notebook de análisis/entrenamiento y la aplicación web Streamlit para consultar el predictor.

## Publicar la aplicación (sin reconstruirla)

1. Sube esta carpeta completa a un repositorio de GitHub.
2. Entra a https://share.streamlit.io/ e inicia sesión con GitHub.
3. Selecciona **Create app** y el repositorio correspondiente.
4. Rama: `main`.
5. Main file path: `app_saber_pro/app.py`.
6. En Advanced settings selecciona Python **3.13** (la versión indicada por el README de la aplicación).
7. Pulsa **Deploy** y espera a que termine la instalación.

El enlace generado será el que se comparte con el profesor. La aplicación necesita los archivos `.joblib`, el JSON de XGBoost y `features_saber_pro.py`, que están junto a `app.py` en `app_saber_pro/`.

## Ejecutar localmente

```bash
cd app_saber_pro
pip install -r requirements.txt
streamlit run app.py
```

## Contenido

- `ISI_Parcial.ipynb`: notebook del proyecto.
- `Saber_Pro_muestra_20pct_multianual.csv`: conjunto de datos utilizado en el análisis.
- `app_saber_pro/`: interfaz, código auxiliar, modelos entrenados y dependencias para ejecutar Streamlit.

**Nota:** el notebook y el CSV se conservan como material del proyecto; la aplicación publicada utiliza los modelos ya entrenados y no necesita volver a ejecutar el notebook.
