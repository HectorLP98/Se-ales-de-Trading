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