import os
import sys
import pandas as pd
#import datetime
import streamlit as st
from binance.spot import Spot
from binance.client import Client
from datetime import datetime, timedelta

from GraficasPloty import * #graficar_velas, graficar_linea, 
from Std_streamlit import  * 



# Configuración general
st.set_page_config(layout="wide", page_title="Análisis Técnico")

# Insert your Binance API key and secret
API_KEY = 'tTH25XYPnXjPQnOfTHbcd8y6FGaq6QXyIUf7jbR1iisyebqq5KByMyg8KFNHgn3h'
API_SECRET = 'fgPIxiygL5QvE5cPmopmCemxUYKQz2ThrAzFMEXz7nlPyeMcMaYnSrCHsyq62dAL'

mercado = "Spot" # Futuros, Spot

if mercado == "Spot":
    # Initialize Binance client
    client = Client(API_KEY, API_SECRET)
    # Obtener toda la información del exchange
    exchange_info = client.get_exchange_info()
    # Filtrar solo los símbolos activos del mercado spot
    simbolos = [
        s["symbol"]
        for s in exchange_info["symbols"]
        if s["status"] == "TRADING" and s["isSpotTradingAllowed"]
    ]
    
else:
    from binance.um_futures import UMFutures
    # Usa tus claves API si es necesario
    client = UMFutures(key=API_KEY, secret=API_SECRET)
    exchange_info_futures = client.exchange_info()
    # Filtrar solo los símbolos activos
    simbolos = [
        s["symbol"]
        for s in exchange_info_futures["symbols"]
        if s["status"] == "TRADING"
    ]




#simbolos = [x for x in simbolos if str(x).find("USD")>=0]
#simbolos = simbolos[:10]
indicadores_disponibles = ["Media_Movil", "Bandas Bollinger", "RSI", "MACD", "Estocastico", "Calidad PullBack"]

# Sidebar
with st.sidebar:
    st.title("📊 Parámetros de Análisis")

    # Intervalo único
    interval = st.selectbox("Intervalo", (
        "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1W", "1M"), index=10)

    # Fecha de inicio
    st.markdown("### Rango temporal de Análisis")
    periodos = st.number_input("Periodos a descargar", min_value=1, max_value=10000, value=90)
    periodicidad = st.selectbox("Periodicidad", ["minutos", "horas", "días", "semanas"], index=2)

    delta_dict = {
        "minutos": timedelta(minutes=periodos),
        "horas": timedelta(hours=periodos),
        "días": timedelta(days=periodos),
        "semanas": timedelta(weeks=periodos)
    }

    fecha_inicio = datetime.now() - delta_dict[periodicidad]
    fecha_inicio = fecha_inicio.strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"🕒 Fecha de inicio calculada: **{fecha_inicio}**")

    # Rango de análisis
    excluir_final = st.number_input("Periodos a excluir", min_value=1, max_value=10000, value=7)
    rango_analisis = st.number_input("Periodos a analizar", min_value=1, max_value=10000, value=60)
    rango_analisis = (excluir_final, rango_analisis)
    
    # Total de simbolos a buscar
    n_sim = st.number_input("Cuantos simbolos desea descargar?", min_value=3,max_value=len(simbolos), value=len(simbolos), help="Ideal para probar unos pocos simbolos tus cambios, de lo contrario descarga todo")
    
# Validaciones de acceso a Binance
try:
    cliente = Spot(key="3bTMORx0HLuEpVuqn3gBgYzupfOVKLPS2QkFazAsFP2sLg0ktAFxbkZa76aQ4VTv", secret="tCNjEOdRFyvlKEH9usVsclEs0izi623zU26nWQRND1Huny1zqQdJ9vBQu4etfrKg")
except Exception as e:
    st.error("🔐 Error en autenticación con Binance. Verifica tus credenciales.")
    st.stop()

# Construir nombre del archivo CSV con sufijo de intervalo
interval_suffix = interval.lower() if isinstance(interval, str) else "unknown"
nombre_archivo_csv = f"/home/hector/Documentos/Escuelas/Autodidacta/Git_repositories/Trading_test/Data/Analisis_simbolos_{interval_suffix}.csv"




# Cargar archivo si existe
if os.path.exists(nombre_archivo_csv):
    # Obtener timestamp de última modificación
    timestamp_modificacion = os.path.getmtime(nombre_archivo_csv)

    # Convertir a fecha legible
    ultima_modificacion = datetime.fromtimestamp(timestamp_modificacion).strftime("%Y-%m-%d %H:%M:%S")

    # Mostrar la métrica
    st.metric(
        label="📁 Nombre del archivo",
        value=nombre_archivo_csv
    )

    df_resultado = pd.read_csv(nombre_archivo_csv)
    st.success(f"Datos cargados desde {nombre_archivo_csv}")
else:
    st.warning("⚠️ No se ha generado aún el archivo de resultados. Por favor, ejecuta el análisis.")
    
parametros_indicadores = Solicita_Parametros_Indicadores()

# Mostrar qué seleccionó el usuario
if parametros_indicadores:
    #print(indicadores_seleccionados)
    st.markdown(f"**Indicadores seleccionados:** {', '.join(parametros_indicadores.keys())}")
    
# Botón para ejecutar análisis nuevamente
if st.button("🔄 Ejecutar análisis"):
    with st.spinner("Procesando símbolos..."):
        try:
            df_resultado = analizar_simbolos(simbolos[:n_sim], cliente, interval, fecha_inicio, rango_analisis=rango_analisis, dict_parametros=parametros_indicadores)
            if not df_resultado.empty:
                # Guardar con sufijo y fecha
                fecha_actual = datetime.now().strftime("%Y%m%d_%H%M")
                nombre_guardado = f"analisis_simbolos_{interval_suffix}_{fecha_actual}.csv"
                df_resultado.to_csv(nombre_archivo_csv, index=False)
                print(f"✅ Análisis completado. Archivo guardado como: `{nombre_guardado}`")
                st.success(f"✅ Análisis completado. Archivo guardado como: `{nombre_guardado}`")

            else:
                st.warning("⚠️ El análisis no arrojó resultados.")
        except Exception as e:
            st.error(f"❌ Error durante el análisis: {e}")

# Mostrar filtros y tabla si hay resultados cargados o recién generados
if 'df_resultado' in locals() and not df_resultado.empty:

    st.markdown("### 🔍 Filtros adicionales")
    patron = st.text_input("🔎 Filtrar por patrón en símbolo (ej: USD)", value="USD")
    estrategia = st.selectbox("Estrategia", ["","MM_Estocastico"], index=0)
    c1, c2, c3 = st.columns(3)
    
    #ascendente = st.checkbox("Orden ascendente", value=False)

    # Aplicar filtros
    df_filtrado = df_resultado.copy()
    if patron.strip() != "":
        df_filtrado = df_filtrado[df_filtrado["Simbolo"].str.contains(patron, case=False, na=False)]
        
    if estrategia == "":
        columnas = seleccionar_columnas(df_filtrado)
        df_filtrado = df_filtrado[columnas]
        filtros, ordenamientos = solicitar_filtros_orden(df_filtrado)
        df_filtrado = aplicar_filtros_orden(df_filtrado, filtros, ordenamientos)


    elif estrategia == "MM_Estocastico":
        ver_tendencia = c1.selectbox("Tendencia", ["Alcista","Bajista"], index=0, help="Filtra por tipo de tendencia")
        tipo_pull = c2.selectbox("Tipo Pull Back", ["Alcista","Bajista"], index=1, help="Filtra por tipo de tendencia")
        duracionTrend = c3.number_input("Duracion Pull back",min_value=1, value=3, help="Cuanto es la duracion maxima que se muestra")
        mostrarPctj = c1.selectbox("Columna de porcentaje", ["Diferencia_Porcentajes","Porcentaje_Alcista","Porcentaje_Bajista"], index=1, help="Muestra el tipo de columna que eligas")
        tipo_corta,tipo_larga,periodo_corta,periodo_larga = parametros_indicadores["Media_Movil"]
        rename = {f"DuracionTrendMM_Close_{tipo_larga}_{periodo_larga}": "Duracion_Tendencia",
                  f"DuracionTrendMM_Close_{tipo_corta}_{periodo_corta}": "Duracion_PullBack",
                  "Tendencia_MM":"Tendencia", f"Tendencia_PullBack":"Tipo_PullBack",
                  "Estocastico_Senal":"Senal_Estocastico",
                  }
        df_filtrado.rename(columns=rename, inplace=True)
        col_interes = ["Simbolo", "N", mostrarPctj, "%_Cumplimiento_FVG", "Ultimo_FVG_Falta", "Tipo_Vela",
                       "Tendencia","Duracion_Tendencia","Tipo_PullBack","Duracion_PullBack","Senal_Estocastico"]
        
        df_filtrado = df_filtrado[col_interes] [
                                  (df_filtrado["Duracion_PullBack"].astype(int)<=duracionTrend) &
                                  (df_filtrado["Tendencia"].astype(str) !="Neutro") &
                                  (df_filtrado["Tipo_PullBack"]==tipo_pull) & 
                                  (df_filtrado["Senal_Estocastico"].astype(str) !="Neutro") &
                                  (df_filtrado["Tendencia"]==ver_tendencia)
        ]
    

    st.markdown(f"### 📈 Resultados filtrados: {len(df_filtrado)} símbolos encontrados")
    st.metric(
        label="🕒 Última actualización",
        value=ultima_modificacion
    )
    #st.write(df_filtrado)
    st.dataframe(df_filtrado)

    # Botón de descarga
    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar resultados filtrados CSV", data=csv, file_name=f"filtro_{interval_suffix}.csv", mime="text/csv")
