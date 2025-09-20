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
                       "Volumen_acum","Volumen",
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


def filtro_TimeFrame_Mayor(df, dict_parametros, estrategia):
    st.markdown("### 🧭 Filtro de Confirmación de Tendencia (TF Mayor)")
    
    rename = {}
    # Renombrar columnas de interés para todos
    if "Media_Movil" in dict_parametros.keys():
        tipo_corta, tipo_larga, periodo_corta, periodo_larga = dict_parametros["Media_Movil"]
        rename[f"DuracionTrendMM_Close_{tipo_larga}_{periodo_larga}"] = "Duracion_Tendencia"
        rename[f"DuracionTrendMM_Close_{tipo_corta}_{periodo_corta}"] = "Duracion_PullBack"
        rename["Tendencia_MM"] = "Tendencia"
        rename["Tendencia_PullBack"] = "Tipo_PullBack"
    if "Estocastico" in dict_parametros.keys():
        rename["Estocastico_Senal"] = "Senal_Estocastico"
        
    df.rename(columns=rename, inplace=True)
    
    if estrategia == "MM_Estocastico":
        col_interes = ["Simbolo", "N", "Diferencia_Porcentajes", "Porcentaje_Alcista", "Porcentaje_Bajista",
                       "%_Cumplimiento_FVG", "Ultimo_FVG_Falta", "Tipo_Vela",
                       "Tendencia","Duracion_Tendencia","Tipo_PullBack","Duracion_PullBack","Senal_Estocastico"]
        # Filtro para confirmar tendencia alcista o bajista corta
        # Duracion de tendencia 
        c1, c2 = st.columns(2)
        tipo_pull = c1.selectbox("Tipo Pull Back", ["Alcista", "Bajista"], index=1)
        #Tendencia_Max = c1.number_input("Duración Mínima de Tendencia", min_value=1, value=5, help="Cuanto es la duracion minima que se muestra")
        PullBack_Max = c2.number_input("Duración Máxima de Pullback", min_value=1, value=10, help="Cuanto es la duracion maxima que se muestra")
        ver_tendencia = c1.selectbox("Tendencia", ["Alcista", "Bajista",""], index=0)
        df = df[col_interes][
            ((df["Tendencia"] == "Alcista") | (df["Tendencia"] == "Bajista")) & 
            # (df["Duracion_Tendencia"].astype(int) > Tendencia_Max) &
            (df["Duracion_PullBack"].astype(int) <= PullBack_Max) &
            (df["Tipo_PullBack"] == tipo_pull) &
            (df["Tendencia"] == ver_tendencia) &
            (df["Senal_Estocastico"].astype(str) != "Neutro") 
        ]
    if estrategia == "Mini_Max":
        c1, c2 = st.columns(2)
        st.markdown("#### Filtros de Timeframe Mayor")
        #mostrarPctj = c1.selectbox("Columna de porcentaje", ["Diferencia_Porcentajes","Porcentaje_Alcista","Porcentaje_Bajista"], index=1, help="Muestra el tipo de columna que eligas")
        mostrarPctj = "Porcentaje_Alcista"
        col_interes = [
            "Simbolo","N",mostrarPctj, "Cantidad_FVGs","%_Cumplimiento_FVG","Ultimo_FVG_Falta",
            "Total_Swing_Alcistas","Total_Swing_Bajistas","Tendencia",
            "Duracion_Tendencia","Tipo_PullBack","Duracion_PullBack","Duracion_Estocastico","Senal_Estocastico",
        ]
        pctj_min = c1.number_input("Porcentaje mínimo TF Mayor", min_value=5, value=10, step=1, help=f"Porcentaje mínimo para filtrar ({np.percentile(df[mostrarPctj], 25)}, {np.percentile(df[mostrarPctj], 50)}, {np.percentile(df[mostrarPctj], 75)}, {np.percentile(df[mostrarPctj], 100)})", )
        df = df[col_interes][
            (df[mostrarPctj] >= pctj_min) &
            (df["Senal_Estocastico"].astype(str) != "Neutro" )
            ]
        df.sort_values(mostrarPctj, ascending=False, inplace=True)
    return df

def filtro_TimeFrame_Menor(df,dict_parametros, estrategia):
    st.markdown("### 🧭 Filtro de Confirmación de Tendencia (TF Menor)")
    rename = {}
    # Renombrar columnas de interés para todos
    if "Media_Movil" in dict_parametros.keys():
        tipo_corta, tipo_larga, periodo_corta, periodo_larga = dict_parametros["Media_Movil"]
        rename[f"DuracionTrendMM_Close_{tipo_larga}_{periodo_larga}"] = "Duracion_Tendencia"
        rename[f"DuracionTrendMM_Close_{tipo_corta}_{periodo_corta}"] = "Duracion_PullBack"
        rename["Tendencia_MM"] = "Tendencia"
        rename["Tendencia_PullBack"] = "Tipo_PullBack"
    if "Estocastico" in dict_parametros.keys():
        rename["Estocastico_Senal"] = "Senal_Estocastico"
        
    df.rename(columns=rename, inplace=True)
    
    if estrategia == "MM_Estocastico":
        col_interes = ["Simbolo", "N", "Diferencia_Porcentajes", "Porcentaje_Alcista", "Porcentaje_Bajista",
                       "%_Cumplimiento_FVG", "Ultimo_FVG_Falta", "Tipo_Vela",
                       "Tendencia","Duracion_Tendencia","Tipo_PullBack","Duracion_PullBack","Senal_Estocastico"]
        
        c1, c2, c3 = st.columns(3)
        #ver_tendencia = c1.selectbox("Tendencia", ["Alcista", "Bajista",""], index=0)
        
        duracionTrend = c3.number_input("Duración Máxima de Pullback", min_value=1, value=10)

        df = df[col_interes][
            (df["Duracion_PullBack"].astype(int) <= duracionTrend) &
            #(df["Tendencia"].astype(str) != "Neutro") &
            #(df["Tipo_PullBack"] == tipo_pull) &
            (df["Senal_Estocastico"].astype(str) != "Neutro") 
            #(df["Tendencia"] == ver_tendencia)
        ]
    elif estrategia == "Mini_Max":
        col_interes = [
            "Simbolo","N","Cantidad_FVGs","%_Cumplimiento_FVG","Ultimo_FVG_Falta",
            "Total_Swing_Alcistas","Total_Swing_Bajistas","Tendencia",
            "Duracion_Tendencia","Tipo_PullBack","Duracion_PullBack","Duracion_Estocastico","Senal_Estocastico",
        ]
        c1, c2 = st.columns(2)
        #mostrarPctj = c1.selectbox("Columna de porcentaje", ["Diferencia_Porcentajes","Porcentaje_Alcista","Porcentaje_Bajista"], index=1, help="Muestra el tipo de columna que eligas")
        mostrarPctj = "Porcentaje_Alcista"
        pctj_min = c1.number_input("Porcentaje mínimo TF Menor", min_value=5, value=10, step=1, help=f"Porcentaje mínimo para filtrar ({np.percentile(df[mostrarPctj], 25)}, {np.percentile(df[mostrarPctj], 50)}, {np.percentile(df[mostrarPctj], 75)}, {np.percentile(df[mostrarPctj], 100)})", )
        col_interes.append(mostrarPctj)
        df = df[col_interes][df[mostrarPctj] >= pctj_min]
        df.sort_values(mostrarPctj, ascending=False, inplace=True)
    return df
    
    
def filtrar_por_estrategia(df: pd.DataFrame, config: dict, parametros_indicadores: dict) -> pd.DataFrame:
    """Aplica el filtrado según 'config' a un DataFrame.
    - No muta 'df' (evita problemas al filtrar múltiples DataFrames).
    - Tolera columnas faltantes usando intersecciones y máscaras seguras.
    """
    if not config or len(config) != 1:
        raise ValueError("config debe tener exactamente 1 estrategia como clave")

    estrategia, params = next(iter(config.items()))
    # Copia ligera para no tocar el original (evita errores al filtrar varios dfs)
    _df = df.copy(deep=False)
    
    # Cambiamos el nombre de las columnas según la estrategia
    rename = {}
    # Renombrar columnas de interés para todos
    if "Media_Movil" in parametros_indicadores.keys():
        tipo_corta, tipo_larga, periodo_corta, periodo_larga = parametros_indicadores["Media_Movil"]
        rename[f"DuracionTrendMM_Close_{tipo_larga}_{periodo_larga}"] = "Duracion_Tendencia"
        rename[f"DuracionTrendMM_Close_{tipo_corta}_{periodo_corta}"] = "Duracion_PullBack"
        rename["Tendencia_MM"] = "Tendencia"
        rename["Tendencia_PullBack"] = "Tipo_PullBack"
    if "Estocastico" in parametros_indicadores.keys():
        rename["Estocastico_Senal"] = "Senal_Estocastico"
        
    _df.rename(columns=rename, inplace=True)

    if estrategia == "":
        columnas = params.get("columnas", [])
        ordenar_por = params.get("ordenar_por", [])
        ascendente = params.get("ascendente", True)

        if columnas:
            cols = [c for c in columnas if c in _df.columns]
            _df = _df.loc[:, cols]
        if ordenar_por:
            sort_cols = [c for c in ordenar_por if c in _df.columns]
            if sort_cols:
                _df = _df.sort_values(sort_cols, ascending=ascendente)
        return _df

    if estrategia == "MM_Estocastico":
        # Construcción robusta del rename a partir de los parámetros de MM
        base_cols = ["Simbolo", "N", "Diferencia_Porcentajes", "Porcentaje_Alcista", "Porcentaje_Bajista",
                       "%_Cumplimiento_FVG", "Ultimo_FVG_Falta", "Tipo_Vela",
                       "Tendencia","Duracion_Tendencia","Tipo_PullBack","Duracion_PullBack","Senal_Estocastico"]
        
        
        col_pct = params.get("mostrarPctj")
        duracionTrend = params.get("duracionTrend", 3)
        ver_tendencia = params.get("ver_tendencia", "Alcista")
        tipo_pull = params.get("tipo_pull", "Alcista")
        
        base_cols.append(col_pct)
        cols_finales = [c for c in base_cols if c in _df.columns]

        _df = _df[base_cols][
            (_df["Duracion_PullBack"].astype(int) <= duracionTrend) &
            (_df["Tendencia"].astype(str) != "Neutro") &
            (_df["Tipo_PullBack"] == tipo_pull) &
            (_df["Senal_Estocastico"].astype(str) != "Neutro") &
            (_df["Tendencia"] == ver_tendencia)
        ]
       
        return _df

    if estrategia == "Mini_Max":
        tipo_operacion = params.get("tipo_operacion", "Alcista")
        porcentaje_min = params.get("porcentaje_min", 25)
        agregar_indicadores = params.get("agregar_indicadores", True)

        col_porc = "Porcentaje_Alcista" if tipo_operacion == "Alcista" else "Porcentaje_Bajista"
        base_cols = ["Simbolo", "N", "%_Cumplimiento_FVG", "Ultimo_FVG_Falta", "Tipo_Vela", "Precio_Actual", col_porc]

        if agregar_indicadores:
            # Añadir solo indicadores que existan en el df
            extra = [k for k in parametros_indicadores.keys() if isinstance(k, str) and k in _df.columns]
            base_cols += extra

        cols_finales = [c for c in base_cols if c in _df.columns]
        mask = pd.Series(True, index=_df.index)
        if col_porc in _df.columns:
            mask &= pd.to_numeric(_df[col_porc], errors='coerce').fillna(-np.inf) > porcentaje_min

        if not cols_finales:
            return _df.loc[mask]
        return _df.loc[mask, cols_finales]

    # Si la estrategia no es reconocida, devolver sin cambios
    return _df
