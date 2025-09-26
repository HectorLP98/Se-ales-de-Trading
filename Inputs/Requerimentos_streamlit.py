import streamlit as st
import pandas as pd

def ui_solicitar_estrategia(df_muestra: pd.DataFrame) -> dict:
    """UI en Streamlit para elegir estrategia y capturar sus parámetros.
    Retorna un diccionario con la clave = nombre de la estrategia y el valor = params.
    Ejemplo de retorno: {"MM_Estocastico": {"ver_tendencia": "Alcista", ...}}
    """
    estrategias = ["", "MM_Estocastico", "Mini_Max"]
    estrategia = st.selectbox("Estrategia", estrategias, index=1)

    st.markdown("### Parámetros de la Estrategia")
    if estrategia == "":
        # Modo libre: elegir columnas y orden
        columnas = st.multiselect("Columnas a mostrar", options=list(df_muestra.columns))
        ordenar_por = st.multiselect("Ordenar por", options=columnas)
        ascendente = st.checkbox("Ascendente", value=True)
        return {"": {"columnas": columnas, "ordenar_por": ordenar_por, "ascendente": ascendente}}

    if estrategia == "MM_Estocastico":
        c1, c2, c3 = st.columns(3)
        ver_tendencia = c1.selectbox("Tendencia", ["Alcista", "Bajista"], index=0)
        tipo_pull = c2.selectbox("Tipo Pull Back", ["Alcista", "Bajista"], index=1)
        duracionTrend = int(c3.number_input("Duración Pull back", min_value=1, value=3))
        mostrarPctj = c1.selectbox("Columna de porcentaje", [
            "Diferencia_Porcentajes", "Porcentaje_Alcista", "Porcentaje_Bajista"
        ], index=1)
        return {"MM_Estocastico": {
            "ver_tendencia": ver_tendencia,
            "tipo_pull": tipo_pull,
            "duracionTrend": duracionTrend,
            "mostrarPctj": mostrarPctj,
        }}

    if estrategia == "Mini_Max":
        c1, c2 = st.columns(2)
        tipo_operacion = c1.selectbox("Tipo de operación", ["Alcista", "Bajista"], index=0)
        porcentaje_min = int(c2.number_input("Porcentaje mínimo", min_value=0, value=25, step=1))
        agregar_indicadores = c1.checkbox("Agregar indicadores", value=True)
        return {"Mini_Max": {
            "tipo_operacion": tipo_operacion,
            "porcentaje_min": porcentaje_min,
            "agregar_indicadores": agregar_indicadores,
        }}

    # Fallback (no debería ocurrir)
    return {"": {}}

def Solicita_Parametros_Indicadores():
    parametros_indicadores = {}
    
    # Selector múltiple de indicadores técnicos
    indicadores_disponibles = ["Media_Movil", "Bandas Bollinger", "RSI", "MACD", "Estocastico", "Calidad_PullBack"]

    indicadores = st.multiselect(
        "📊 Selecciona los indicadores técnicos a mostrar:",
        options=indicadores_disponibles,
        default=["Media_Movil","Estocastico" ],
        help="Puedes elegir más de un indicador."
    )

    if "Media_Movil" in indicadores:
        st.markdown(f"### Medias Moviles")
        c1, c2 = st.columns(2)
        tipo_corta = c1.selectbox(
            "Tipo de media corta",
            ["EMA", "SMA", "WMA"],
            index=0,
            help="Selecciona el tipo de media móvil para el periodo corto."
        )

        tipo_larga = c2.selectbox(
            "Tipo de media larga",
            ["EMA", "SMA", "WMA"],
            index=1,
            help="Selecciona el tipo de media móvil para el periodo largo."
        )

        periodo_corta = c1.number_input(
            "Periodo media corta",
            min_value=1,
            max_value=500,
            value=20,
            help="Define cuántos periodos usará la media móvil corta."
        )

        periodo_larga = c2.number_input(
            "Periodo media larga",
            min_value=1,
            max_value=500,
            value=200,
            help="Define cuántos periodos usará la media móvil larga."
        )
        parametros_indicadores["Media_Movil"] = (tipo_corta, tipo_larga, periodo_corta, periodo_larga)
    if "Estocastico" in indicadores:
        st.markdown(f"### Estocastico")
        c1, c2 = st.columns(2)
        n = c1.number_input("n:", value=14, help="Es el período usado para calcular %K.")
        d = c2.number_input("d:", value=3, help="Es el período de suavizado de %K para calcular %D.")
        parametros_indicadores["Estocastico"] = (n,d)
    if "Calidad_PullBack" in indicadores:
        st.markdown(f"### Calidad Pull Back")
        c1, c2 = st.columns(2)
        window = c1.number_input("Window Cruces", min_value=5, value=20, help="Toma como referencia temporal, cuenta la cantidad de cruces el ultimos w-periodos")
        tresh = c2.number_input("Treshold rango", min_value=0.00001, value=0.2, help="Este sirve para medir que tanto se ha movido sginificativamente con respecto a la vela anterior hace_: rango = (h - l)/l*100 if rango < threshold:")
        parametros_indicadores["IQP"] = [window, tresh]
        
    if "RSI" in indicadores:
        st.markdown(f"### RSI")
        per = st.number_input("Periodos", help="Cantidad de periodos que se usan para calcualr el RSI",
                        min_value=3,value=14)
        parametros_indicadores["RSI"] = per
        
    return parametros_indicadores