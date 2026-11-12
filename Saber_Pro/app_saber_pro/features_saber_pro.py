# ============================================================
# Variables derivadas del proyecto Saber Pro
# ============================================================
# Este archivo se usa en el notebook Y en la app de Streamlit. Los Pipelines guardados lo referencian
# por nombre, así que debe estar en la misma carpeta que app.py.
# Solo usa las 19 variables originales de UN estudiante (fila a fila): no aprende nada de los datos,
# por lo que no puede producir fuga de información.
import numpy as np
import pandas as pd

BIENES = ['FAMI_TIENEAUTOMOVIL', 'FAMI_TIENELAVADORA', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET']
PAGOS = ['ESTU_PAGOMATRICULABECA', 'ESTU_PAGOMATRICULACREDITO',
         'ESTU_PAGOMATRICULAPADRES', 'ESTU_PAGOMATRICULAPROPIO']

RANGO_EDUCACION = {'Ninguno': 0, 'Primaria incompleta': 1, 'Primaria completa': 2,
                   'Secundaria (Bachillerato) incompleta': 3, 'Secundaria (Bachillerato) completa': 4,
                   'Técnica o tecnológica incompleta': 5, 'Técnica o tecnológica completa': 6,
                   'Educación profesional incompleta': 7, 'Educación profesional completa': 8, 'Postgrado': 9}
RANGO_MATRICULA = {'No pagó matrícula': 0, 'Menos de 500 mil': 1, 'Entre 500 mil y menos de 1 millón': 2,
                   'Entre 1 millón y menos de 2.5 millones': 3, 'Entre 2.5 millones y menos de 4 millones': 4,
                   'Entre 4 millones y menos de 5.5 millones': 5, 'Entre 5.5 millones y menos de 7 millones': 6,
                   'Más de 7 millones': 7}
RANGO_HORAS = {'No trabaja': 0, 'Menos de 10 horas': 1, 'Entre 11 y 20 horas': 2,
               'Entre 21 y 30 horas': 3, 'Más de 30 horas': 4}

NUM_DERIVADAS = ['N_BIENES', 'N_FUENTES_PAGO', 'EDU_MAX_PADRES', 'EDU_MIN_PADRES', 'ESTRATO_NUM',
                 'HORAS_NUM', 'MATRICULA_NUM', 'EDAD_DIF25_CUAD']
CAT_DERIVADAS = ['EDAD_RANGO', 'DEPTO_x_ORIGEN', 'CARACTER_x_ORIGEN', 'ORIGEN_x_METODO',
                 'ORIGEN_x_MATRICULA', 'DEPTO_x_ESTRATO']


def _es_si(serie):
    t = serie.astype(str).str.strip().str.lower().str.replace('í', 'i', regex=False)
    return (t == 'si').astype(float)


def _cruce(a, b):
    return a.astype(str) + ' | ' + b.astype(str)


def agregar_variables(X):
    """Devuelve una copia de X con las variables derivadas añadidas (las 19 originales se conservan)."""
    X = X.copy()
    X['N_BIENES'] = sum(_es_si(X[c]) for c in BIENES)
    X['N_FUENTES_PAGO'] = sum(_es_si(X[c]) for c in PAGOS)

    padre = X['FAMI_EDUCACIONPADRE'].map(RANGO_EDUCACION).astype(float)
    madre = X['FAMI_EDUCACIONMADRE'].map(RANGO_EDUCACION).astype(float)
    X['EDU_MAX_PADRES'] = np.fmax(padre, madre)
    X['EDU_MIN_PADRES'] = np.fmin(padre, madre)

    X['ESTRATO_NUM'] = X['FAMI_ESTRATOVIVIENDA'].astype(str).str.extract(r'(\d)')[0].astype(float)
    X['HORAS_NUM'] = X['ESTU_HORASSEMANATRABAJA'].map(RANGO_HORAS).astype(float)
    X['MATRICULA_NUM'] = X['ESTU_VALORMATRICULAUNIVERSIDAD'].map(RANGO_MATRICULA).astype(float)

    edad = pd.to_numeric(X['EDAD'], errors='coerce')
    X['EDAD_DIF25_CUAD'] = (edad - 25) ** 2          # efecto no lineal de la edad (para el modelo lineal)
    rango = pd.cut(edad, [0, 21, 24, 28, 35, 120], labels=['<=21', '22-24', '25-28', '29-35', '>35'])
    X['EDAD_RANGO'] = rango.astype(object).fillna('Sin información')

    # Cruces: el efecto de una variable cambia según otra (p. ej., la región depende del tipo de institución)
    X['DEPTO_x_ORIGEN'] = _cruce(X['ESTU_DEPTO_RESIDE'], X['INST_ORIGEN'])
    X['CARACTER_x_ORIGEN'] = _cruce(X['INST_CARACTER_ACADEMICO'], X['INST_ORIGEN'])
    X['ORIGEN_x_METODO'] = _cruce(X['INST_ORIGEN'], X['ESTU_METODO_PRGM'])
    X['ORIGEN_x_MATRICULA'] = _cruce(X['INST_ORIGEN'], X['ESTU_VALORMATRICULAUNIVERSIDAD'])
    X['DEPTO_x_ESTRATO'] = _cruce(X['ESTU_DEPTO_RESIDE'], X['FAMI_ESTRATOVIVIENDA'])
    return X
