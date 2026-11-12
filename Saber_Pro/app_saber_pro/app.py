# ============================================================
# 🎓 Saber Pro · Predictor de rendimiento académico (Streamlit)
# ============================================================
# Ejecutar:  streamlit run app.py
# Necesita en la misma carpeta: features_saber_pro.py, modelos_saber_pro.joblib, xgboost_modelo.json y resultados_saber_pro.joblib
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Saber Pro · Predictor de rendimiento", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

try:
    import features_saber_pro  # noqa: F401  (los Pipelines guardados lo referencian por nombre)
except ImportError:
    st.error("Falta el archivo `features_saber_pro.py`. Cópialo en la misma carpeta que `app.py`.")
    st.stop()

AZUL, VERDE, ROJO, AMBAR = "#1f4e79", "#1e9e6a", "#d64545", "#e0a100"
COLORES_MODELO = ["#1f4e79", "#e07b39", "#2a9d8f", "#8e44ad"]

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.hero {background: linear-gradient(120deg, #1f4e79 0%, #2f7fb8 55%, #3fa9d6 100%); color: white;
       padding: 1.4rem 1.8rem; border-radius: 18px; margin-bottom: 1.1rem;
       box-shadow: 0 10px 28px rgba(31,78,121,.25);}
.hero h1 {color: white; font-size: 1.85rem; margin: 0; font-weight: 800;}
.hero p {color: #e6f1fb; margin: .35rem 0 0 0; font-size: 1rem;}
.chip {display: inline-block; background: rgba(255,255,255,.18); color: white; padding: .18rem .7rem;
       border-radius: 999px; font-size: .78rem; margin: .55rem .35rem 0 0;}
.result {border-radius: 18px; padding: 1.4rem 1.5rem; color: white; text-align: center;
         box-shadow: 0 8px 22px rgba(0,0,0,.12);}
.result h2 {color: white; margin: 0; font-size: 1.75rem; font-weight: 800;}
.result p {margin: .35rem 0 0 0; color: rgba(255,255,255,.93); font-size: .98rem;}
.nota {background: #f3f6fa; border-left: 4px solid #2f7fb8; padding: .75rem 1rem; border-radius: 8px;
       color: #374151; font-size: .9rem;}
div[data-testid="stMetric"] {background: rgba(47,127,184,.06); border: 1px solid rgba(47,127,184,.18);
       border-radius: 14px; padding: .7rem .9rem;}
.stTabs [data-baseweb="tab-list"] {gap: .35rem;}
.stTabs [data-baseweb="tab"] {background: rgba(47,127,184,.08); border-radius: 10px 10px 0 0; padding: .5rem 1rem;}
.stTabs [aria-selected="true"] {background: #1f4e79 !important; color: white !important;}
.foot {color: #9ca3af; font-size: .8rem; text-align: center; margin-top: 2.2rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Carga de modelos y resultados
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos…")
def cargar():
    modelos = {}
    for n, m in joblib.load(BASE_DIR / "modelos_saber_pro.joblib").items():
        # XGBoost viene en formato JSON nativo (portable entre sistemas operativos)
        if isinstance(m, dict) and m.get("tipo") == "xgboost_json":
            clf = XGBClassifier()
            archivo_modelo = Path(m["archivo"])
            if not archivo_modelo.is_absolute():
                archivo_modelo = BASE_DIR / archivo_modelo
            clf.load_model(str(archivo_modelo))
            m = Pipeline([("prep", m["prep"]), ("clf", clf)])
        modelos[n] = m
    return modelos, joblib.load(BASE_DIR / "resultados_saber_pro.joblib")


try:
    MODELOS, R = cargar()
except FileNotFoundError:
    st.error("No se encontraron los archivos de modelos. Copia `modelos_saber_pro.joblib`, "
             "`xgboost_modelo.json`, `resultados_saber_pro.joblib` y `features_saber_pro.py` en la misma carpeta que `app.py`.")
    st.stop()

VARS = R["variables"]
PREDICTORAS = R["predictoras"]
NOMBRES = list(MODELOS)
ETIQ = {v["col"]: v["etiqueta"] for v in VARS}
FORMATO = {"F": "Femenino", "M": "Masculino"}
GRUPOS = [("👤", "Estudiante"), ("🏠", "Hogar y familia"), ("💳", "Financiación"), ("🏛️", "Institución")]
Y_TEST = np.asarray(R["y_test"])


def fmt(x):
    return FORMATO.get(x, x)


def clasificar(p, u):
    return "alto" if p >= u else "bajo"


def fig_matriz(cm, titulo, escala="Blues"):
    total = cm.sum()
    nombres = [["TN", "FP"], ["FN", "TP"]]
    texto = [[f"<b>{nombres[i][j]}</b><br>{cm[i, j]:,}<br>({cm[i, j] / total * 100:.1f} %)" for j in range(2)]
             for i in range(2)]
    fig = go.Figure(go.Heatmap(z=cm, x=["Bajo (0)", "Alto (1)"], y=["Bajo (0)", "Alto (1)"], text=texto,
                               texttemplate="%{text}", textfont={"size": 15}, colorscale=escala,
                               showscale=False, hoverinfo="skip"))
    fig.update_layout(title=titulo, height=360, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="Predicción", yaxis_title="Real", yaxis_autorange="reversed")
    return fig


# ------------------------------------------------------------
# Barra lateral
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Saber Pro ML")
    st.caption("Clasificación de rendimiento académico (alto vs. bajo) a partir de 19 variables "
               "sociodemográficas, socioeconómicas, de financiación e institucionales.")
    modelo = st.selectbox("Modelo", NOMBRES, index=NOMBRES.index(R["seleccionado"]),
                          format_func=lambda n: f"{n}  🏆" if n == R["seleccionado"] else n)
    umbral = float(R["umbrales"][modelo])   # punto de corte elegido con validación cruzada
    st.divider()
    st.markdown("**Convención**  \n🔴 0 = Bajo rendimiento  \n🟢 1 = Alto rendimiento")
    st.caption("Elige el modelo con el que quieres hacer las predicciones y ver su desempeño. "
               "🏆 marca el modelo con mejor desempeño en validación.")


st.markdown(f"""
<div class="hero">
  <h1>🎓 Predictor de rendimiento en Saber Pro</h1>
  <p>Explora cómo se asocian las características de un estudiante con su probabilidad de pertenecer al
     grupo de alto rendimiento (≥ P80 de su año) o bajo rendimiento (≤ P20).</p>
  <span class="chip">📚 {R['n_total']:,} estudiantes · 2018–2022</span>
  <span class="chip">🤖 {len(NOMBRES)} modelos</span>
  <span class="chip">🏆 Mejor: {R['seleccionado']}</span>
  <span class="chip">🧪 Prueba: {R['n_test']:,} estudiantes</span>
</div>""", unsafe_allow_html=True)

tab_pred, tab_eval, tab_imp, tab_info = st.tabs(
    ["🔮 Predicción individual", "📊 Desempeño de modelos", "🔎 Importancia de variables",
     "ℹ️ Acerca del proyecto"])

# ============================================================
# 1. PREDICCIÓN INDIVIDUAL
# ============================================================
with tab_pred:
    st.markdown("#### Perfil del estudiante")
    with st.form("formulario_perfil"):
        valores = {}
        for icono, grupo in GRUPOS:
            vars_g = [v for v in VARS if v["grupo"] == grupo]
            st.markdown(f"**{icono} {grupo}**")
            cols = st.columns(3)
            for i, v in enumerate(vars_g):
                with cols[i % 3]:
                    if v["tipo"] == "numerica":
                        valores[v["col"]] = st.number_input(v["etiqueta"], int(v["min"]), int(v["max"]),
                                                            int(v["defecto"]), 1)
                    else:
                        valores[v["col"]] = st.selectbox(v["etiqueta"], v["opciones"],
                                                         index=v["opciones"].index(v["defecto"]), format_func=fmt)
        enviado = st.form_submit_button("🔮 Calcular predicción", type="primary", use_container_width=True)

    if enviado:
        st.session_state["perfil_estudiante"] = valores

    if "perfil_estudiante" not in st.session_state:
        st.info("Completa el perfil y pulsa **Calcular predicción**. Los valores iniciales corresponden "
                "a la categoría más frecuente de cada variable.")
    else:
        perfil = pd.DataFrame([st.session_state["perfil_estudiante"]])[PREDICTORAS]
        probs = {n: float(m.predict_proba(perfil)[:, 1][0]) for n, m in MODELOS.items()}
        p = probs[modelo]
        clase = clasificar(p, umbral)
        estilo = {"alto": (VERDE, "🟢 Alto rendimiento", "El perfil se parece más al grupo ≥ P80."),
                  "bajo": (ROJO, "🔴 Bajo rendimiento", "El perfil se parece más al grupo ≤ P20.")}[clase]

        c1, c2 = st.columns([1, 1.15])
        with c1:
            st.markdown(f"""<div class="result" style="background:{estilo[0]}">
                <h2>{estilo[1]}</h2><p>{estilo[2]}</p>
                <p style="margin-top:.8rem;font-size:1.05rem"><b>{modelo}</b> · Probabilidad de alto rendimiento:
                <b>{p*100:.1f} %</b></p></div>""", unsafe_allow_html=True)
            st.write("")
            comp = pd.DataFrame({"Modelo": list(probs), "P(alto)": [probs[n] for n in probs]})
            comp["Clasificación"] = [
                ("🟢 Alto" if clasificar(probs[n], R["umbrales"][n]) == "alto" else "🔴 Bajo") for n in probs]
            st.dataframe(comp.style.format({"P(alto)": "{:.1%}"}), hide_index=True, use_container_width=True)
        with c2:
            pasos = [{"range": [0, umbral * 100], "color": "#fbe3e3"},
                     {"range": [umbral * 100, 100], "color": "#dff5ea"}]
            g = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": " %"},
                                       title={"text": "Probabilidad de alto rendimiento"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": estilo[0]},
                                              "steps": pasos,
                                              "threshold": {"line": {"color": "#111827", "width": 4},
                                                            "value": umbral * 100}}))
            g.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(g, use_container_width=True)

        # Barras comparando los 4 modelos
        fb = go.Figure()
        for i, n in enumerate(probs):
            fb.add_bar(y=[n], x=[probs[n] * 100], orientation="h", marker_color=COLORES_MODELO[i % 4],
                       text=f"{probs[n]*100:.1f} %", textposition="outside", showlegend=False)
        fb.add_vline(x=50, line_dash="dot", line_color="#9ca3af")
        fb.update_layout(title="¿Qué dice cada modelo?", xaxis_title="P(alto) %", xaxis_range=[0, 105],
                         height=260, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fb, use_container_width=True)

        # Análisis "¿qué pasaría si…?"
        st.markdown("#### 🧭 ¿Qué cambios moverían más la probabilidad?")
        filas, etiquetas = [], []
        for v in VARS:
            actual = st.session_state["perfil_estudiante"][v["col"]]
            alternativas = (list(range(int(v["min"]), int(v["max"]) + 1, 2)) if v["tipo"] == "numerica"
                            else [o for o in v["opciones"] if o not in (actual, "Sin información")])
            for alt in alternativas:
                fila = dict(st.session_state["perfil_estudiante"])
                fila[v["col"]] = alt
                filas.append(fila)
                etiquetas.append((v["etiqueta"], alt))
        if filas:
            p_alt = MODELOS[modelo].predict_proba(pd.DataFrame(filas)[PREDICTORAS])[:, 1]
            efectos = pd.DataFrame(etiquetas, columns=["variable", "valor"])
            efectos["cambio"] = (p_alt - p) * 100
            mejor = efectos.loc[efectos.groupby("variable")["cambio"].apply(lambda s: s.abs().idxmax())]
            mejor = mejor.reindex(mejor["cambio"].abs().sort_values(ascending=False).index).head(8)
            mejor["texto"] = [f"{r.variable} → {fmt(r.valor)}" for r in mejor.itertuples()]
            fw = go.Figure(go.Bar(y=mejor["texto"][::-1], x=mejor["cambio"][::-1], orientation="h",
                                  marker_color=[VERDE if c > 0 else ROJO for c in mejor["cambio"][::-1]],
                                  text=[f"{c:+.1f} pp" for c in mejor["cambio"][::-1]], textposition="outside"))
            fw.update_layout(height=380, margin=dict(l=10, r=40, t=10, b=10),
                             xaxis_title="Cambio en P(alto), puntos porcentuales")
            st.plotly_chart(fw, use_container_width=True)
        st.markdown('<div class="nota">⚠️ Estos cambios muestran <b>asociaciones aprendidas por el modelo</b>, '
                    'no efectos causales: cambiar una característica en la vida real no garantiza el mismo '
                    'cambio en el rendimiento.</div>', unsafe_allow_html=True)

# ============================================================
# 2. DESEMPEÑO DE MODELOS
# ============================================================
with tab_eval:
    p_test = np.asarray(R["prob_test"][modelo])
    cm = confusion_matrix(Y_TEST, (p_test >= umbral).astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_clas = cm.sum()

    st.markdown(f"#### {modelo} en el conjunto de prueba ({n_clas:,} estudiantes)")
    k = st.columns(6)
    k[0].metric("Accuracy", f"{(tp + tn) / n_clas:.3f}")
    k[1].metric("Recall (alto)", f"{tp / (tp + fn):.3f}")
    k[2].metric("Especificidad (bajo)", f"{tn / (tn + fp):.3f}")
    k[3].metric("ROC-AUC", f"{R['metricas'].loc[modelo, 'ROC-AUC']:.3f}")
    k[4].metric("Falsos positivos (FP)", f"{fp:,}")
    k[5].metric("Falsos negativos (FN)", f"{fn:,}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_matriz(cm, f"Matriz de confusión · {modelo}"), use_container_width=True)
        st.markdown('<div class="nota"><b>FP</b>: el modelo dijo <i>alto</i> pero era de bajo rendimiento '
                    '(estudiante en riesgo no detectado).<br><b>FN</b>: el modelo dijo <i>bajo</i> pero era de '
                    'alto rendimiento.</div>', unsafe_allow_html=True)
    with c2:
        fr = go.Figure()
        for i, n in enumerate(NOMBRES):
            pn = np.asarray(R["prob_test"][n])
            f_, t_, _ = roc_curve(Y_TEST, pn)
            auc = R["metricas"].loc[n, "ROC-AUC"]
            fr.add_scatter(x=f_, y=t_, name=f"{n} (AUC {auc:.3f})",
                           line=dict(color=COLORES_MODELO[i % 4], width=3.5 if n == modelo else 1.6))
        fr.add_scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(color="gray", dash="dash"))
        fr.update_layout(title="Curvas ROC (los 4 modelos)", xaxis_title="Tasa de falsos positivos",
                         yaxis_title="Sensibilidad", height=420, margin=dict(l=10, r=10, t=50, b=10),
                         legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fr, use_container_width=True)

    st.markdown("#### Comparación de los 4 modelos")
    fe = go.Figure()
    for col, color in [("FP", ROJO), ("FN", AZUL)]:
        fe.add_bar(x=NOMBRES, y=[R["metricas"].loc[n, col] for n in NOMBRES], name=col, marker_color=color,
                   text=[f"{int(R['metricas'].loc[n, col]):,}" for n in NOMBRES], textposition="outside")
    fe.update_layout(barmode="group", title="Errores por modelo", yaxis_title="Estudiantes", height=380,
                     margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fe, use_container_width=True)
    tabla = R["metricas"].drop(columns=["umbral"], errors="ignore").copy()
    cols_num = ["Accuracy", "Precision", "Recall", "Especificidad", "F1", "ROC-AUC"]
    st.dataframe(tabla.style.format({c: "{:.4f}" for c in cols_num if c in tabla})
                 .format({c: "{:,.0f}" for c in ["TN", "FP", "FN", "TP", "FP + FN"] if c in tabla})
                 .highlight_min(subset=["FP + FN"], color="#d1fae5"),
                 use_container_width=True)

# ============================================================
# 3. IMPORTANCIA DE VARIABLES
# ============================================================
with tab_imp:
    st.markdown("#### Permutation importance")
    st.caption("Cuánto cae el ROC-AUC en prueba cuando se desordena cada variable. Cuanto mayor la caída, "
               "más depende el modelo de esa variable para distinguir los grupos.")
    elegidos = st.multiselect("Modelos a comparar", NOMBRES, default=[modelo])
    imp = R["importancias"].copy()
    if elegidos:
        orden = imp[elegidos].mean(axis=1).sort_values().index
        fi = go.Figure()
        for i, n in enumerate(elegidos):
            fi.add_bar(y=[ETIQ.get(c, c) for c in orden], x=imp.loc[orden, n], name=n, orientation="h",
                       marker_color=COLORES_MODELO[NOMBRES.index(n) % 4])
        fi.update_layout(barmode="group", height=640, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="Caída del ROC-AUC", legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fi, use_container_width=True)
    st.markdown('<div class="nota">La importancia es <b>predictiva</b>, no causal. Variables asociadas entre sí '
                '(p. ej., estrato y automóvil) se reparten la información, así que su importancia individual '
                'puede verse menor.</div>', unsafe_allow_html=True)

# ============================================================
# 4. ACERCA DEL PROYECTO
# ============================================================
with tab_info:
    st.markdown(f"""
#### Metodología
- **Datos:** resultados Saber Pro 2018–2022 (muestra del 20 %), {R['n_total']:,} estudiantes en los extremos de la distribución.
- **Variable objetivo:** por año, **1 = alto** (puntaje promedio ≥ P80) y **0 = bajo** (≤ P20). Se excluye la zona intermedia.
- **Predictores:** 19 variables sociodemográficas, socioeconómicas, laborales, de financiación e institucionales.
  El puntaje y todo lo derivado de él **no** se usan como predictores. Con esas 19 respuestas el modelo calcula internamente variables derivadas
  (bienes del hogar, educación máxima de los padres, rango de edad y cruces como departamento × origen de la institución).
- **Validación:** división estratificada 80/20 ({R['n_train']:,} / {R['n_test']:,}); hiperparámetros con validación cruzada de 5 particiones;
  punto de corte de cada modelo elegido con validación cruzada, sin usar el conjunto de prueba.
- **Modelos (uno por familia):** Regresión Logística (lineal; modelo de la entrega anterior), Decision Tree (árbol individual),
  Random Forest (*bagging*) y XGBoost (*boosting*).

#### Limitaciones
- Las importancias y los análisis "¿qué pasaría si…?" muestran **asociaciones**, no causas.
- El modelo distingue los extremos (P20 vs P80); no está pensado para clasificar a estudiantes del grupo intermedio.
- El archivo original solo trae el promedio, así que no se pudieron detectar los estudiantes con un único módulo en 0.
""")
    st.markdown("#### Cortes de la variable objetivo por año")
    st.dataframe(R["cortes"], use_container_width=True)

st.markdown('<div class="foot">Proyecto académico · Introducción a los Sistemas Inteligentes · UAJS</div>',
            unsafe_allow_html=True)
