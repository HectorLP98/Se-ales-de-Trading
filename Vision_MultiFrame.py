import os
import sys
import pandas as pd
#import datetime
import streamlit as st
from binance.spot import Spot
from binance.client import Client
from datetime import datetime, timedelta


# Añadir rutas personalizadas
sys.path.append('/home/hector/Documentos/Escuelas/Autodidacta/Git_repositories/Trading_test')


from GraficasPloty import * #graficar_velas, graficar_linea, 
from Std_streamlit import  * 



# Configuración general
st.set_page_config(layout="wide", page_title="Analisis MultiFrame")

# Insert your Binance API key and secret
API_KEY = 'tTH25XYPnXjPQnOfTHbcd8y6FGaq6QXyIUf7jbR1iisyebqq5KByMyg8KFNHgn3h'
API_SECRET = 'fgPIxiygL5QvE5cPmopmCemxUYKQz2ThrAzFMEXz7nlPyeMcMaYnSrCHsyq62dAL'

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Obtener toda la información del exchange
exchange_info = client.get_exchange_info()

# Filtrar solo los símbolos activos del mercado spot
simbolos_spot = [
    s["symbol"]
    for s in exchange_info["symbols"]
    if s["status"] == "TRADING" and s["isSpotTradingAllowed"]
]
simbolos_spot = [x for x in simbolos_spot if str(x).find("USD")>=0]
#simbolos_spot = simbolos_spot[:10]
indicadores_disponibles = ["Media_Movil", "Bandas Bollinger", "RSI", "MACD", "Estocastico", "Calidad PullBack"]

# Validaciones de acceso a Binance
try:
    cliente = Spot(key="3bTMORx0HLuEpVuqn3gBgYzupfOVKLPS2QkFazAsFP2sLg0ktAFxbkZa76aQ4VTv", secret="tCNjEOdRFyvlKEH9usVsclEs0izi623zU26nWQRND1Huny1zqQdJ9vBQu4etfrKg")
except Exception as e:
    st.error("🔐 Error en autenticación con Binance. Verifica tus credenciales.")
    st.stop()
    
    
    
# Sidebar
with st.sidebar:
    dict_grl_frame = obtener_parametros_globales_MF()
    timeframes = dict_grl_frame.keys()

st.write(dict_grl_frame)
st.markdown(f"**Timeframes a analizar:** {', '.join(timeframes)}")

# Seleccion de indicadores
parametros_indicadores = Solicita_Parametros_Indicadores()

# Diccionario con los DataFrames
dataframes_por_tf = {}

# Intentar cargar todos los archivos primero
for tf in timeframes:
    df, path, fecha_mod = cargar_df_existente( tf)
    if df is not None:
        dataframes_por_tf[f"df_{tf}"] = df

# Forzar actualización manual si ya hay datos cargados
if dataframes_por_tf:
    forzar_actualizacion_dataframes(simbolos_spot[:5], dict_grl_frame, cliente, parametros_indicadores)
    
    
# Si faltan archivos, ofrecer generar solo los que faltan
faltantes = [tf for tf in timeframes if f"df_{tf}" not in dataframes_por_tf]

if faltantes:
    st.warning(f"⚠️ No hay datos para: {', '.join(faltantes)}")
    if st.button("🔄 Ejecutar análisis SOLO para los faltantes"):
        with st.spinner("Descargando y generando análisis..."):
            for tf in faltantes:
                _, path, _ = cargar_df_existente(simbolo, tf)  # solo para obtener el path
                df = generar_df_nuevo(simbolo, tf, fecha_inicio, rango_analisis, parametros_indicadores, cliente, path)
                if not df.empty:
                    dataframes_por_tf[f"df_{tf}"] = df

# Toma solo uno de los DataFrames (por ejemplo el primero)
df_referencia = list(dataframes_por_tf.values())[0]

# Solicita Filtros
columnas = seleccionar_columnas(df_referencia)
filtros, ordenamientos = solicitar_filtros_orden(df_referencia)


def filtro_TimeFrame_Mayor(df, dict_parametros):
    st.markdown("### 🧭 Filtro de Confirmación de Tendencia (TF Mayor)")
    # Renombrar columnas de interés para todos
    tipo_corta, tipo_larga, periodo_corta, periodo_larga = dict_parametros["Media_Movil"]
    rename = {
                f"DuracionTrendMM_Close_{tipo_larga}_{periodo_larga}": "Duracion_Tendencia",
                f"DuracionTrendMM_Close_{tipo_corta}_{periodo_corta}": "Duracion_PullBack",
                "Tendencia_MM": "Tendencia",
                "Tendencia_PullBack": "Tipo_PullBack",
                "Estocastico_Senal": "Senal_Estocastico",
            }
    df.rename(columns=rename, inplace=True)

    # Filtro para confirmar tendencia alcista o bajista corta
    df = df[
        (df["Tendencia"] == "Alcista") |
        ((df["Tendencia"] == "Bajista") & (df["Duracion_Tendencia"].astype(int) < 5))
    ]
    return df

def filtro_TimeFrame_Menor(df_filtrado,dict_parametros):
    st.markdown("### 🧭 Filtro de Confirmación de Tendencia (TF Menor)")
    # Renombrar columnas de interés para todos
    tipo_corta, tipo_larga, periodo_corta, periodo_larga = dict_parametros["Media_Movil"]
    rename = {
                f"DuracionTrendMM_Close_{tipo_larga}_{periodo_larga}": "Duracion_Tendencia",
                f"DuracionTrendMM_Close_{tipo_corta}_{periodo_corta}": "Duracion_PullBack",
                "Tendencia_MM": "Tendencia",
                "Tendencia_PullBack": "Tipo_PullBack",
                "Estocastico_Senal": "Senal_Estocastico",
            }
    df_filtrado.rename(columns=rename, inplace=True)

    c1, c2, c3 = st.columns(3)
    ver_tendencia = c1.selectbox("Tendencia", ["Alcista", "Bajista"], index=0)
    tipo_pull = c2.selectbox("Tipo Pull Back", ["Alcista", "Bajista"], index=1)
    duracionTrend = c3.number_input("Duración Máxima de Pullback", min_value=1, value=3)

    df_filtrado = df_filtrado[
        (df_filtrado["Duracion_PullBack"].astype(int) <= duracionTrend) &
        (df_filtrado["Tendencia"].astype(str) != "Neutro") &
        (df_filtrado["Tipo_PullBack"] == tipo_pull) &
        (df_filtrado["Senal_Estocastico"].astype(str) != "Neutro") &
        (df_filtrado["Tendencia"] == ver_tendencia)
    ]
    return df_filtrado
    

if dataframes_por_tf:
    tabs = st.tabs([f"{tf}" for tf in dataframes_por_tf.keys()] + ["Compare"])
    
    for i, (key, df) in enumerate(dataframes_por_tf.items()):
        tf_actual = key.replace("df_", "").lower()
        
        with tabs[i]:
        
            st.subheader(f"📈 Resultados para `{tf_actual}`")
            col_interes = [
                "Simbolo", "N", "Diferencia_Porcentajes", "Porcentaje_Alcista", "Porcentaje_Bajista",
                "%_Cumplimiento_FVG", "Ultimo_FVG_Falta", "Tipo_Vela",
                "Tendencia", "Duracion_Tendencia", "Tipo_PullBack", "Duracion_PullBack", "Senal_Estocastico"
            ]

            if tf_actual in ["4h", "1d"]:

                # Filtro para confirmar tendencia alcista o bajista corta
                df_filtrado = filtro_TimeFrame_Mayor(df, parametros_indicadores)

            else:
                df_filtrado = filtro_TimeFrame_Menor(df, parametros_indicadores)

            st.dataframe(df_filtrado[col_interes].tail())

    with tabs[-1]:
        # Seleccion de time Frame mayor
        mayor = st.selectbox("TimeFrame Mayor", dataframes_por_tf.keys())
        # Seleccion de time Frame menor
        menor = st.selectbox("TimeFrame Menor", dataframes_por_tf.keys())
        st.write(menor, mayor)