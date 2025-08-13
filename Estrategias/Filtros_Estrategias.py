import pandas as pd 
import streamlit as st
from Std_streamlit import  * 

def Filtrar_x_estrategia(estrategia, parametros_indicadores, df_filtrado):
        
    if estrategia == "":
        columnas = seleccionar_columnas(df_filtrado)
        df_filtrado = df_filtrado[columnas]
        filtros, ordenamientos = solicitar_filtros_orden(df_filtrado)
        df_filtrado = aplicar_filtros_orden(df_filtrado, filtros, ordenamientos)
        
    elif estrategia == "MM_Estocastico":
        c1, c2, c3 = st.columns(3)
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
    elif estrategia == "Mini_Max":
        c1, c2 = st.columns(2)
        col_interes = ["Simbolo", "N", "%_Cumplimiento_FVG", "Ultimo_FVG_Falta", "Tipo_Vela",
                       "Precio_Actual"]
        tipo_operacion = c1.selectbox("Tipo de operación", ["Alcista","Bajista"], index=0, help="Filtra por tipo de operación")
        porcentaje_min = c2.number_input("Porcentaje mínimo", min_value=5, value=25, step=1, help="Porcentaje mínimo para filtrar")
        if tipo_operacion=="Alcista":
            col_porc = "Porcentaje_Alcista"
        else:
            col_porc = "Porcentaje_Bajista"
            
        
        agregar_indicadores = c1.checkbox("Agregar indicadores", value=True, help="Agrega indicadores de tendencia y volumen")
        if agregar_indicadores:
            col_interes += parametros_indicadores.keys()
        
        col_interes += [col_porc]
        df_filtrado = df_filtrado[col_interes][
                                    (df_filtrado[col_porc] > porcentaje_min) 
        ]
        
    return df_filtrado