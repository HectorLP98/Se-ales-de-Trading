import os
import sys
import pandas as pd
import streamlit as st
from binance.spot import Spot as Client
#from binance.client import Client
from datetime import datetime, timedelta
from Inputs.Simbolos import Obtener_Simbolos_Binance
from Inputs.Requerimentos_streamlit import Solicita_Parametros_Indicadores
from GraficasPloty import * #graficar_velas, graficar_linea, 
from Std_streamlit import  * 
from Estrategias.Filtros_Estrategias import *


# Configuración general
st.set_page_config(layout="wide", page_title="Vision General")

# Insert your Binance API key and secret
API_KEY = 'z0wEy09JKt0gzqzBUmAUdlEgP83fgQUOUKmOdWURp3aI3qVVCfAVi3QrEowwjzNg'
API_SECRET =  '4y26OxaTdtdcQMKmJuNt1446vh9k4Iyi9y6nayueVCeOHEUi8qppW7ioN9WJdJXd'
ruta_data = r"./Datos/Historicos"

mercado = "Spot" # Futuros, Spot

if mercado == "Spot":
    # Validaciones de acceso a Binance
    try:
        cliente = Client(key=API_KEY, secret=API_SECRET)
    except Exception as e:
        st.error("🔐 Error en autenticación con Binance. Verifica tus credenciales.")
        st.stop()
    
else:
    from binance.um_futures import UMFutures
    # Usa tus claves API si es necesario
    cliente = UMFutures(key=API_KEY, secret=API_SECRET)
    exchange_info_futures = cliente.exchange_info()
    # Filtrar solo los símbolos activos
    simbolos = [
        s["symbol"]
        for s in exchange_info_futures["symbols"]
        if s["status"] == "TRADING"
    ]


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
    
    c1, c2 = st.columns(2)
    mercado = c1.selectbox("Mercado", ("Spot", "Futuros"), index=0)
      
    patron = c2.text_input("🔎 Filtrar por patrón en símbolo (ej: USD)", value="USDT")
    if patron.strip() == "":
        patron = None
    simbolos = Obtener_Simbolos_Binance(cliente,mercado=mercado,filtro=patron)
    
    # Total de simbolos a buscar
    n_sim = c1.number_input("Cuantos simbolos desea descargar?", min_value=3,max_value=len(simbolos), value=len(simbolos), help=f"Ideal para probar unos pocos simbolos tus cambios, de lo contrario descarga todo (Hay {len(simbolos)} simbolos disponibles)")
    


# Construir nombre del archivo CSV con sufijo de intervalo
interval_suffix = interval.lower() if isinstance(interval, str) else "unknown"
nombre_archivo_csv = f"{ruta_data}/Analisis_simbolos_{interval_suffix}.csv"




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
            error_trace = traceback.format_exc()  # Obtiene toda la traza del error
            st.error(f"❌ Error al durante al procesar los simbolos en Vision_General:\n{error_trace}")

# Mostrar filtros y tabla si hay resultados cargados o recién generados
if 'df_resultado' in locals() and not df_resultado.empty:

    st.markdown("### 🔍 Filtros adicionales")
    estrategia = st.selectbox("Estrategia", ["","MM_Estocastico","Mini_Max"], index=0)
    
    
    #ascendente = st.checkbox("Orden ascendente", value=False)

    # Aplicar filtros
    df_filtrado = df_resultado.copy()
    
    tab1, tab2, tab3, tab4 = st.tabs([ "🗃 Datos", "📈 Chart","📊 Estadísticas", "📉 Correlación"])
    
    with tab1:
        # Datos
        df_filtrado = Filtrar_x_estrategia(estrategia, parametros_indicadores, df_filtrado)
        st.markdown(f"### 📈 Resultados filtrados: {len(df_filtrado)} símbolos encontrados")
        st.metric(
            label="🕒 Última actualización",
            value=ultima_modificacion
        )
        #st.write(df_filtrado)
        st.dataframe(df_filtrado)

    with tab2:# Mostrar gráficos en Streamlit
        top = st.number_input("Cuántos símbolos mostrar en gráficos?", min_value=1, max_value=50, value=20)
        fig, simbolo_bar = plot_top_bottom_volumen(df_filtrado, col_use="Volumen_acum", top_n=top)
        st.pyplot(fig)
        
        # Ejemplo de uso en streamlit
        fig, simbol_heatmap = plot_volumen_heatmap(df_filtrado, col_use="Volumen", top_n=top)
        st.plotly_chart(fig, use_container_width=True)
        # Ejemplo de uso
        # Ejemplo: mostrar burbujas con volumen normal
        inner_simbol = set(simbolo_bar).intersection(set(simbol_heatmap))
        fig = plot_volumen_bubbles(df_filtrado[df_filtrado["Simbolo"].isin(inner_simbol)], col_use="Volumen", top_n=top)
        st.plotly_chart(fig, use_container_width=True)


    # Botón de descarga
    csv = df_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar resultados filtrados CSV", data=csv, file_name=f"filtro_{interval_suffix}.csv", mime="text/csv")
