import tqdm
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import sys 
from datetime import datetime, timedelta
import os
import traceback
from Indicadores.Direccion import Medias_Moviles
from Datos.Datasets import Get_Sets_Hist
#      from fn_prep import get_change_percent
from Indicadores.Direccion import Medias_Moviles, calcular_MACD
from Indicadores.Momento import *
from Indicadores.Volatilidad import calcular_bandas_bollinger
from Indicadores.DominioPropio import *
from Estrategias.Busqueda_entry import *


ruta_data = r"./Datos/Historicos"

def diccionario_TimeFrames(interval):
    equivalencias = {
        "1M": ("1M", "1W"),
        "1W": ("1W", "4h"),
        "3d": ("3d", "1d"),
        "1d": ("1d", "1h"),
        "12h": ("12h", "1h"),
        "8h": ("8h", "1h"),
        "6h": ("6h", "30m"),
        "4h": ("4h", "15m"),
        "2h": ("2h", "15m"),
        "1h": ("1h", "5m"),
        "30m": ("30m", "5m"),
        "15m": ("15m", "1m"),
        "5m": ("5m", "1m"),
        "1m": ("1m", "1m"),
    }

    return equivalencias.get(interval, (interval, interval))

def preparar_dataset(cliente, symbol, interval, fecha_inicio, use_binance, periodos_ema=[7, 29]):
    try:
        if isinstance(fecha_inicio, (int, float)):
            periodos = int(fecha_inicio)
        elif isinstance(fecha_inicio, str):
            periodos = str(fecha_inicio)
        df = Get_Sets_Hist(cliente=cliente, symbol=symbol, interval=interval, periodos=periodos, use_binance=use_binance)
        if df.empty:
            st.warning(f"⚠️ Dataset vacío para {symbol} en intervalo {interval}.")
            return None

        df = df.set_index("ct")

        # Limpieza de columnas irrelevantes
        if "Ignore" in df.columns:
            df.drop(columns='Ignore', inplace=True)
        df.columns = df.columns.get_level_values(0)

        # Clasificación de velas
        df["tipo"] = df.apply(lambda row: "long" if row["Open"] < row["Close"]
                              else "neutral" if row["Close"] == row["Open"] else "short", axis=1)


        return df

    except Exception as e:
        error_trace = traceback.format_exc()  # Obtiene toda la traza del error
        st.error(f"❌ Error al preparar dataset para {symbol} en intervalo {interval}:\n{error_trace}")

def formatear_diccionario_resultados(diccionario):
    """
    Transforma un diccionario con arrays en un DataFrame horizontal:
    Cada clave se convierte en columnas prefijadas.
    Las filas corresponden a observaciones sincronizadas.

    Args:
        diccionario (dict): Diccionario con arrays de resultados.

    Returns:
        pd.DataFrame: DataFrame con todas las columnas combinadas.
    """
    lista_dfs = []

    for clave, array in diccionario.items():
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        # Crear DataFrame con nombres prefijados
        df_temp = pd.DataFrame(array)
        df_temp.columns = [
            f"{clave}_Col{i+1}" for i in range(df_temp.shape[1])
        ]

        lista_dfs.append(df_temp)

    # Concatenar por columnas
    df_final = pd.concat(lista_dfs, axis=1)

    return df_final.reset_index(drop=False).rename(columns={'index':'orden'})

class AnalizadorIndicadores:
    def __init__(self, df: pd.DataFrame,  dict_parametros:dict):
        """
        Inicializa el AnalizadorIndicadores.
        
        indicadores: lista de indicadores a calcular.
        df: DataFrame original.
        """
        self.indicadores = dict_parametros.keys()
        self.df = df.copy()
        self.df_resultado = df.copy()
        self.dict_parametros = dict_parametros
        self.ult_value_indicator = {}

    def obtener_tendencias_previas(self, col_filtro, cols_retain):
        """
        Devuelve los registros anteriores a los cruces detectados en col_filtro
        más el último registro actual, incluyendo solo las columnas deseadas.

        Args:
            df (pd.DataFrame): DataFrame de entrada.
            col_filtro (str): Columna donde detectar los cruces (valores distintos de 0).
            cols_retain (list): Columnas que quieres conservar.

        Returns:
            list: Lista de diccionarios con los valores extraídos.
        """
        # Posiciones de índices donde hay cruce
        pos_cruces = self.df_resultado.index[self.df_resultado[col_filtro] != 0].tolist()

        # Convertimos a posiciones numéricas (por si el índice es datetime)
        posiciones = [
            self.df_resultado.index.get_loc(i) 
            for i in pos_cruces 
            if self.df_resultado.index.get_loc(i) > 0
        ]

        # Posiciones anteriores
        prev_posiciones = [i - 1 for i in posiciones] + [-1]

        # Extraemos las filas de interés
        df_prev_trends = self.df_resultado.iloc[prev_posiciones, :][cols_retain]

        # Convertimos a lista de dicts (puedes cambiar por df_prev_trends si prefieres DataFrame)
        lista_resultados = df_prev_trends.values

        return lista_resultados[:]


    def determinar_tendencia_mm(self, row, tipo_larga, periodo_larga):
        """
        Determina la tendencia según la media móvil larga.
        """
        valor_media = row[f"{tipo_larga}_{periodo_larga}"]
        if pd.isna(valor_media) or pd.isna(row["Close"]):
            return "Neutro"
        if row["Close"] > valor_media:
            return "Alcista"
        elif row["Close"] <= valor_media:
            return "Bajista"
        else:
            return "Neutro"
    
    def Genera_Indicadores(self):
        """
        Genera indicadores en base a la lista de indicadores seleccionados.
        """
        
        # Creamos la columna de entrada y salida
        #self.df_resultado["Entrada_Trade"] = greedy_trades(self.df_resultado["Close"].values)
        #self.df_resultado["Entrada_Trade"] = greedy_trades_tp_sl(self.df_resultado["Close"], tp=0.03, sl=0.01, t_max=30)
        self.df_resultado["Entrada_Trade"] = greedy_trades_trailing(self.df_resultado["Close"], tp=0.10, sl=0.03, trailing=0.05)
        
        if "Media_Movil" in self.indicadores:
            # Parametros
            self.tipo_corta,self.tipo_larga,self.periodo_corta,self.periodo_larga = self.dict_parametros["Media_Movil"]
            # Usamos tu clase de medias moviles
            mm = Medias_Moviles()
            self.df_resultado = mm.ColocarMM(self.df_resultado,
                                             columna_precio="Close",
                                             tipo_corta=self.tipo_corta,
                                             periodo_corta=self.periodo_corta,
                                             tipo_larga=self.tipo_larga,
                                             periodo_larga=self.periodo_larga)
            # Generar señal de tendencia vs EMA_larga
            #print(f"{self.tipo_larga}_{self.periodo_larga}")
            self.df_resultado["Tendencia_MM"] = self.df_resultado.apply(
                                                                            self.determinar_tendencia_mm,
                                                                            axis=1,
                                                                            args=(self.tipo_larga, self.periodo_larga)
                                                                        )
            self.df_resultado["Tendencia_PullBack"] = self.df_resultado.apply(
                                                                            self.determinar_tendencia_mm,
                                                                            axis=1,
                                                                            args=(self.tipo_corta, self.periodo_corta)
                                                                        )
            # Generar señal de tendencia vs EMA_larga
            self.df_resultado = mm.calcular_cruce_y_duracion(self.df_resultado, col_corta="Close", col_larga=f"{self.tipo_larga}_{self.periodo_larga}")
            self.df_resultado = mm.calcular_cruce_y_duracion(self.df_resultado, col_corta="Close", col_larga=f"{self.tipo_corta}_{self.periodo_corta}")
            self.ult_value_indicator["MM"] = self.df_resultado.iloc[-1][["Tendencia_MM",f"DuracionTrendMM_Close_{self.tipo_larga}_{self.periodo_larga}",
                                                                         "Tendencia_PullBack",f"DuracionTrendMM_Close_{self.tipo_corta}_{self.periodo_corta}"]].to_dict()
            
        if "Bandas Bollinger" in self.indicadores:
            columna_precio, ventana, num_std = self.dict_parametros["Bandas Bollinger"]
            self.df_resultado = calcular_bandas_bollinger(
                self.df_resultado,
                columna_precio,
                ventana,
                num_std
            )

        if "RSI" in self.indicadores:
            periodos = self.dict_parametros["RSI"]
            self.df_resultado["RSI"] = calcular_rsi(self.df_resultado["Close"], periodos)
            # Generar señal
            self.df_resultado["Señal_RSI"] = generar_senal_rsi(self.df_resultado["RSI"],sobreventa=20, sobrecompra=80)
            self.df_resultado["Duracion_RSI"] = calcular_duracion_senal(self.df_resultado["Señal_RSI"])
            self.ult_value_indicator["RSI"] = self.df_resultado.iloc[-1][["RSI","Señal_RSI","Duracion_RSI"]].to_dict()

        if "Estocastico" in self.indicadores:
            n,d = self.dict_parametros["Estocastico"]
            self.df_resultado = calcular_estocastico(self.df_resultado, n, d)
            self.df_resultado["Estocastico_Senal"] = self.df_resultado.apply(estocastico_signal, axis=1)
            self.df_resultado["Duracion_Estocastico"] = calcular_duracion_estocastico_numpy(self.df_resultado["Estocastico_Senal"].map(lambda x: "-".join(x.split("-")[:-1]) if len(x.split("-"))==3 else x))
            self.ult_value_indicator["Estocastico"] = self.df_resultado.iloc[-1][["%K","%D","Duracion_Estocastico","Estocastico_Senal"]].to_dict()
            
        if "Calidad_PullBack" in self.indicadores:
            window = self.dict_parametros["IQP"][0]
            tresh = self.dict_parametros["IQP"][1]
            self.df_resultado["IQP"] = indice_calidad_pullback(self.df_resultado, window_cruces=window , threshold_rango=tresh, columnas_medias=[f"{self.tipo_corta}_{self.periodo_corta}", f"{self.tipo_corta}_{self.periodo_corta}"])
                        
        if "MACD" in self.indicadores:
            # Usa la clase de medias móviles para calcular MACD
            self.df_resultado = calcular_MACD(self.df_resultado, col_use="Close")
            self.ult_value_indicator["MACD"] = self.df_resultado.iloc[-1][["Signal_Line","Histograma_MACD"]].to_dict()

        return self.df_resultado

    def Probabilidad_duracion_tendencia(self, duraciones):
        # Separar
        alcistas = duraciones[::2]
        bajistas = duraciones[1::2]

        # Ratios
        ratios = [a/b if b !=0 else np.nan for a,b in zip(alcistas, bajistas)]

        # Estadísticas básicas
        media_alcista = np.mean(alcistas)
        media_bajista = np.mean(bajistas)

        # Probabilidad empírica: Alcista > Bajista
        prob_alcista_mas_larga = np.mean([a>b for a,b in zip(alcistas, bajistas)])

        # Probabilidad: Alcista >6 periodos
        prob_alcista_gt6 = np.mean([a>6 for a in alcistas])

        # Autocorrelación entre duraciones consecutivas
        duraciones_pares = np.array(alcistas + bajistas)
        autocorrelacion = np.corrcoef(duraciones_pares[:-1], duraciones_pares[1:])[0,1]

        # Resultados
        resumen = {
            "Media Alcista": media_alcista,
            "Media Bajista": media_bajista,
            "Prob Alcista > Bajista": prob_alcista_mas_larga,
            "Prob Alcista >6": prob_alcista_gt6,
            "Autocorrelacion": autocorrelacion,
            "Ratios": ratios
        }

        return resumen
            
    
    def Resumir_tendencia(self, resultados_tendencia):
        #print("--------------",resultados_tendencia)
        duraciones = [v[1] for v in resultados_tendencia]
        # Separar duraciones por tipo de tendencia
        if len(resultados_tendencia[0]) == 2:
            alcistas = [dur for tipo, dur in resultados_tendencia if tipo == 'Alcista']
            bajistas = [dur for tipo, dur in resultados_tendencia if tipo == 'Bajista']
        elif len(resultados_tendencia[0]) == 4:
            alcistas = [dur for tipo, dur,_,_ in resultados_tendencia if tipo == 'Alcista']
            bajistas = [dur for tipo, dur,_,_ in resultados_tendencia if tipo == 'Bajista']
        else:
            raise IndexError(f"Esta queriendo resumir informacion de tendencia con formato no valido, {len(resultados_tendencia)} -- {resultados_tendencia}")

        #dict_proba = self.Probabilidad_duracion_tendencia(duraciones)
        # Calcular estadísticas
        resumen_tendencias = {
            'Alcista': {
                'cantidad': len(alcistas),
                'duraciones': alcistas,
                'promedio_duracion': np.mean(alcistas),
                'desviacion_estandar': np.std(alcistas, ddof=0)
            },
            'Bajista': {
                'cantidad': len(bajistas),
                'duraciones': bajistas,
                'promedio_duracion': np.mean(bajistas),
                'desviacion_estandar': np.std(bajistas, ddof=0)
            },
            #"Probabilidades": dict_proba

        }
        return resumen_tendencias

        
    def Estrategia_MM_Estocastico(self):
        """
        Genera los análisis de tendencias según la estrategia solicitada.
        
        1) Identificas las ultimas 3 tendencias y la duracion de ellas tomando Precio_Cierre vs EMA_larga ej [("Alcista",20), ("Bajista",50),("Alcista",6)], el orden es del mas reciente, 
            hasta el mas alejado en el  tiempo.
        2) Identificas las ultimas 6 tendencias, la duracion de ellas tomando Precio_Cierre vs EMA_corta y la señal del Estocastico, junto con la duracion que tiene:. 
            ej [("Alcista",5, "sobre-venta", 4), ("Bajista",50,"neutro", 10),("Alcista",6, "sobre-venta",1)]
            
        El ultimo registro significa que que esta en tendencia Alcista hace 5 periodos (el precio cierre ha estado 5 periodos por encima de la media movil larga) y que el Estocastico ambas lineas en el rango (20, 80) por 10 periodos.
        """
    
        # Validar que se hayan generado antes
        col_mm_larga = f"{self.tipo_larga}_{self.periodo_larga}"
        col_mm_corta = f"{self.tipo_corta}_{self.periodo_corta}"
        if col_mm_larga not in self.df_resultado.columns or col_mm_corta not in self.df_resultado.columns:
            raise ValueError("No se generaron medias móviles. Ejecuta Genera_Indicadores primero.")
        
        # 1) Informacion de la tendencia: Precio vs MM larga
        cols = [
            "Tendencia_MM",
            f"DuracionTrendMM_Close_{self.tipo_larga}_{self.periodo_larga}"
        ]
        resultados_tendencia = self.obtener_tendencias_previas(
            col_filtro=f"CruceMM_Close_{self.tipo_larga}_{self.periodo_larga}",
            cols_retain=cols
        )
        
        # 2) Informacion del pull back: Precio vs MM corta & estocastico
        cols = [
            "Tendencia_PullBack",
            f"DuracionTrendMM_Close_{self.tipo_corta}_{self.periodo_corta}",
            "Estocastico_Senal",
            "Duracion_Estocastico"
        ]
        resultados_pb = self.obtener_tendencias_previas(
            col_filtro=f"CruceMM_Close_{self.tipo_corta}_{self.periodo_corta}",
            cols_retain=cols
        )
        
        resumen_Tendencia = self.Resumir_tendencia(resultados_tendencia)
        resumen_PullBack = self.Resumir_tendencia(resultados_pb)
        
        dict_estrategia = {"Tendencia":resultados_tendencia[-1],
                           "Pull_Back":resultados_pb[-1],
                           "Estd_Tendencia":resumen_Tendencia,
                           "Estd_PullBack": resumen_PullBack,
                           #"Indicadores": ult_value_indicator,
                           
                           }
        
        return dict_estrategia


class AnalizadorVelas:
    def __init__(self, df, rango_analisis=(10, 100)):
        self.df_original = df
        self.precio_actual = df["Close"].values[-1]
        self.rango_analisis = rango_analisis
        self.fvg_dict = {}
        self.conteo = {
            "Mechas alcistas": 0,
            "Mechas bajistas": 0,
            "Velas alcistas": 0,
            "Velas bajistas": 0,
            "Iteraciones": 0
        }

    def _filtrar_df(self):
        excluir_ultimas, tomar_n = self.rango_analisis
        df_filtrado = self.df_original.iloc[:-excluir_ultimas] if excluir_ultimas > 0 else self.df_original.copy()
        df_filtrado = df_filtrado.iloc[-tomar_n:] if tomar_n > 0 else df_filtrado
        return df_filtrado

    def _detectar_fvg(self, i):
        secuencial = len(self.fvg_dict)
        if self.arr_low_full[i] > self.arr_high_full[i - 2]:  # FVG alcista
            self.fvg_dict[secuencial] = {
                'fecha_inicio': self.arr_index_full[i - 2],
                'fecha_fin': self.arr_index_full[i],
                'rango': (self.arr_high_full[i - 2], self.arr_low_full[i]),
                'tocada': False,
                'fecha_tocada': None
            }

        elif self.arr_high_full[i] < self.arr_low_full[i - 2]:  # FVG bajista
            self.fvg_dict[secuencial] = {
                'fecha_inicio': self.arr_index_full[i - 2],
                'fecha_fin': self.arr_index_full[i],
                'rango': (self.arr_high_full[i], self.arr_low_full[i - 2]),
                'tocada': False,
                'fecha_tocada': None
            }

    def _contar_rupturas(self, i, min_low, max_high):
        open_ = self.arr_open_full[i]
        close = self.arr_close_full[i]
        high = self.arr_high_full[i]
        low = self.arr_low_full[i]
        tipo_vela = "red" if open_ > close else "green"
        self.conteo["Iteraciones"] += 1

        if high > max_high and ((tipo_vela == "green" and close <= max_high) or (tipo_vela == "red" and open_ <= max_high)):
            self.conteo["Mechas alcistas"] += 1

        elif low < min_low and ((tipo_vela == "red" and close >= min_low) or (tipo_vela == "green" and open_ >= min_low)):
            self.conteo["Mechas bajistas"] += 1

        elif close > max_high or open_ > max_high or high > max_high:
            self.conteo["Velas alcistas"] += 1

        elif close < min_low or open_ < min_low:
            self.conteo["Velas bajistas"] += 1
    
    def _std_max_min(self):
        """
        Calcula el porcentaje de subida y bajada en relación con el precio actual,
        tomando como referencia el máximo y mínimo histórico del DataFrame original.
        """
        df = self.df_original
        precio_actual = df['Close'].iloc[-1]
        maximo_historico = df['High'].max()
        minimo_historico = df['Low'].min()

        porcentaje_alcista = ((maximo_historico - precio_actual) / precio_actual) * 100
        porcentaje_bajista = ((precio_actual - minimo_historico) / precio_actual) * 100

        return {
            "Precio_Actual": precio_actual,
            "Porcentaje_Alcista": round(porcentaje_alcista, 2),
            "Porcentaje_Bajista": round(porcentaje_bajista, 2),
            "Diferencia_Porcentajes": abs(porcentaje_alcista - porcentaje_bajista)
        }
        
    def _verificar_testeo_fvg(self, i):
        fecha_actual = self.arr_index_full[i]
        precio_min = self.arr_low_full[i]
        precio_max = self.arr_high_full[i]

        # Filtrar solo FVGs no tocadas
        fvg_no_tocadas = {k: v for k, v in self.fvg_dict.items() if not v.get("tocada", False)}

        for key, fvg_data in fvg_no_tocadas.items():
            # Solo verificar después de la fecha_fin
            if fecha_actual <= fvg_data["fecha_fin"]:
                continue

            rango_min, rango_max = sorted(fvg_data["rango"])

            # Verificar si hay intersección entre rango de la vela y rango de la FVG
            if max(rango_min, precio_min) <= min(rango_max, precio_max):
                self.fvg_dict[key]["tocada"] = True
                self.fvg_dict[key]["fecha_tocada"] = fecha_actual

    def _resumir_swings(self):
        """
        Detecta swings y devuelve un resumen compacto para el método analizar.
        """
        df = self.df_original.copy()
        df["Swing_Alcista"] = 0
        df["Swing_Bajista"] = 0

        for i in range(1, len(df) - 1):
            if df["Low"].iloc[i] < df["Low"].iloc[i - 1] and df["Low"].iloc[i] < df["Low"].iloc[i + 1]:
                df.at[df.index[i], "Swing_Alcista"] = 1
            if df["High"].iloc[i] > df["High"].iloc[i - 1] and df["High"].iloc[i] > df["High"].iloc[i + 1]:
                df.at[df.index[i], "Swing_Bajista"] = 1

        total_alcistas = df["Swing_Alcista"].sum()
        total_bajistas = df["Swing_Bajista"].sum()

        ult_fecha_alcista = df[df["Swing_Alcista"] == 1].index[-1] if total_alcistas > 0 else None
        ult_fecha_bajista = df[df["Swing_Bajista"] == 1].index[-1] if total_bajistas > 0 else None

        ult_precio_alcista = df.loc[ult_fecha_alcista, "Low"] if ult_fecha_alcista else None
        ult_precio_bajista = df.loc[ult_fecha_bajista, "High"] if ult_fecha_bajista else None

        return {
            "Total_Swing_Alcistas": int(total_alcistas),
            "Total_Swing_Bajistas": int(total_bajistas),
            "Fecha_Ultimo_Swing_Alcista": ult_fecha_alcista,
            "Precio_Ultimo_Swing_Alcista": ult_precio_alcista,
            "Fecha_Ultimo_Swing_Bajista": ult_fecha_bajista,
            "Precio_Ultimo_Swing_Bajista": ult_precio_bajista,
        }

    def _fvg_no_tocadas(self):
        """
        Genera resumen de información de las FVGs no tocadas.
        """
        self.dict_fvg_no_tocadas = {k: v for k,v in self.fvg_dict.items() if isinstance(v, dict) and not v.get("tocada", True)}
        # 1️⃣ Total FVGs detectadas
        total_fvg = sum(1 for k,v in self.fvg_dict.items() if isinstance(v, dict))

        # 2️⃣ Total FVGs no tocadas
        fvg_no_tocadas = {
            k: v for k,v in self.fvg_dict.items()
            if isinstance(v, dict) and not v.get("tocada", True)
        }
        total_no_tocadas = len(fvg_no_tocadas)

        # 3️⃣ Porcentaje tocadas
        pct_tocadas = 0
        if total_fvg > 0:
            pct_tocadas = round(100 * (total_fvg - total_no_tocadas) / total_fvg, 2)

        # 4️⃣ Las últimas 3 FVGs no tocadas con info de tendencia y duración
        # Ordenamos por fecha_fin descendente
        ultimas_no_tocadas = sorted(
            fvg_no_tocadas.items(),
            key=lambda x: x[1]["fecha_fin"],
            reverse=True
        )[:3]

        lista_resumen = []
        for k,v in ultimas_no_tocadas:
            rango_min, rango_max = sorted(v["rango"])
            if self.precio_actual > rango_max:
                tendencia = "Bajista"
            elif self.precio_actual < rango_min:
                tendencia = "Alcista"
            else:
                tendencia = "Dentro"

            # Contar periodos desde fecha_fin hasta último índice del df
            fecha_fin = pd.to_datetime(str(v["fecha_fin"]))
            ultimo_index = self.df_original.index[-1]
            if isinstance(ultimo_index, pd.Timestamp):
                fecha_ultima = ultimo_index
            else:
                fecha_ultima = pd.to_datetime(str(ultimo_index))

            # Contar cuántos registros están entre fecha_fin y fecha_ultima
            periodos = len(self.df_original.loc[
                (self.df_original.index > fecha_fin) &
                (self.df_original.index <= fecha_ultima)
            ])

            lista_resumen.append( (tendencia, periodos) )

        # También devolverlos si los quieres usar
        return {
            "total_fvg": total_fvg,
            "total_no_tocadas": total_no_tocadas,
            "pct_tocadas": pct_tocadas,
            "ultimas_fvg_info": lista_resumen
        }

    def _clasificar_vela(self, i, pt1=0.5, pt2=0.1, pt3=0.4):
        """
        (Function)
            funcion clasifica la vela en funcion de la mechas y el cuerpo de la vela
        (Parameters)
            - row: diccionario o Serie de pandas (con claves: Open, Close, High, Low)
            - open_, high, low, close: valores individuales (si no se pasa row)
            . pt1: Porcion de la mecha mayoritaria (la parte mas grande de una mecha en proporcion al total de la vela) default 0.5, es decir la mecha debe medir al menos la mitad del total de la vela
            - pt2: Porcion del mecha menoritaria o tolerancia. Default 0.05, es decir la mecha minoritaria debe ser a lo mas el 5% del total de la vela.
            - pt3: Porcion de las mechas para considerarse Doji, se parte de una simetria con limite superior. Default .4, quiere decir que el cuerpo de la vela no supera el 20% para ser considerada Doji
        (Return)
            str: 
        """
        def Color_Vela(open_, close):
            if close > open_:
                return 1  # Vela alcista
            elif close < open_:
                return -1  # Vela bajista
            else:
                return 0  # Vela neutral
            
        def extrae_datos_vela(i):
            return (self.arr_open_full[i], self.arr_high_full[i], self.arr_low_full[i], self.arr_close_full[i])
        
        open_, high, low, close = extrae_datos_vela(i)
        open_ant, high_ant, low_ant, close_ant = extrae_datos_vela(i-1)
        
        rango_total = high - low
        if rango_total == 0:
            return "Indeterminada"

        # Porcion Total de la mecha
        pt = high - low
        # Porcion mecha inferior
        pmi = min(close, open_) - low
        # Porcion mecha superior
        pms = high - max(close, open_)
        # Porcion cuerpo de vela
        pcv = abs(close - open_)
        # Color de la vela
        color = Color_Vela(open_, close)
        color_ant = Color_Vela(open_ant, close_ant)
        
        if pt == 0 :
            return "Plana"
        elif pmi >= pt1*pt and pms <= pt2*pt and color==1:
            return "Martillo"
        elif pmi <= pt2 * pt and pms >= pt1*pt and color==1:
            return "Martillo_Inv"
        if pmi >= pt1*pt and pms <= pt2*pt and color==-1:
            return "Hang_Man"
        elif pmi <= pt2 * pt and pms >= pt1*pt and color==-1:
            return "Estrella_Fug"
        elif pmi >= pt3*pt and pms >= pt3*pt:
            return "Doji"
        elif pcv >= 0.95*pt:
            return "Cuerpo_Lleno"
        elif color==1 and color_ant ==-1 and open_ <= close_ant and close >= open_ant:
            return "Engulfing_Alcista"
        elif color==-1 and color_ant==1 and open_ >= close_ant and close <= open_ant:
            return "Engulfing_Bearish"
        elif color==1 and color_ant ==-1 and high <= open_ant and low >= close_ant:
            return "Harami_Alcista"
        elif color==-1 and color_ant==1 and high <= close_ant and low >= open_ant:
            return "Harami_Bearish"
        elif color_ant==1 and open_ > close_ant and pmi >= pt3*pt and pms >= pt3*pt:
            return "Doji start Bearish"
        elif color_ant==-1 and open_ < close_ant and pmi >= pt3*pt and pms >= pt3*pt:
            return "Doji start Bullish"
        else:
            return "N/A"


    def analizar(self):
        dict_std_min_max = self._std_max_min()
        df_filtrado = self._filtrar_df()
        #print(f"Fecha en rango {min(df_filtrado.index)} ---- {max(df_filtrado.index)}")

        self.arr_open_full = self.df_original['Open'].to_numpy()
        self.arr_high_full = self.df_original['High'].to_numpy()
        self.arr_low_full = self.df_original['Low'].to_numpy()
        self.arr_close_full = self.df_original['Close'].to_numpy()
        self.arr_index_full = self.df_original.index.to_numpy()

        arr_high_filtrado = df_filtrado['High'].to_numpy()
        arr_low_filtrado = df_filtrado['Low'].to_numpy()

        min_low = np.min(arr_low_filtrado)
        max_high = np.max(arr_high_filtrado)
        #print("********* Rango análisis: ", min_low, max_high)

        # Encontrar fechas correspondientes al mínimo y máximo
        fecha_min = df_filtrado.loc[df_filtrado['Low'] == min_low].index[0]
        fecha_max = df_filtrado.loc[df_filtrado['High'] == max_high].index[0]

        # Fecha de inicio para contar rupturas: la más reciente entre fecha_min y fecha_max
        fecha_inicio_ruptura = max(fecha_min, fecha_max)

        clasificacion_vela = {}
        # Iterar sobre todo el df_original
        for i in range(2, len(self.df_original)):
            fecha_actual = self.df_original.index[i]

            #if fecha_actual in df_filtrado.index:
            self._detectar_fvg(i)
            self._verificar_testeo_fvg(i)          
            
            if fecha_actual >= fecha_inicio_ruptura:
                self._contar_rupturas(i, min_low, max_high)
                # Clasificar vela
                #clasificacion = self._clasificar_vela(i)
                tipo = self._clasificar_vela(self, i, pt1=0.6, pt2=0.05, pt3=0.4)
                clasificacion_vela[str(fecha_actual)]= {"Vela_Ind":tipo}
        
        # Swing points
        resumen_swings = self._resumir_swings()
        
        # FVGs no tocadas
        dict_fvg_no_tocadas = self._fvg_no_tocadas()
        
        return {
            "FVGs": self.fvg_dict,
            "FVGs_no_tocadas": dict_fvg_no_tocadas,
            "Conteo_Rupturas": self.conteo,
            "Rango_Min_Max": (min_low, max_high),
            "Std_Min_Max" : dict_std_min_max,
            "Resumen_Swings": resumen_swings,
            "Velas": clasificacion_vela,
    }

# Funcion para analizar masivamente
def analizar_simbolos(simbolos_spot, cliente, interval, fecha_inicio, rango_analisis=(10, 50), indicadores=[], dict_parametros={}):
    import traceback
    resultados = []
    if len(simbolos_spot) <150: update_interval = 1
    else:    update_interval = int(len(simbolos_spot)/100)
    # Barra de progreso y contenedor de resultados
    progress_bar = st.progress(0)
    results_container = st.empty()
    st.write(f"Entro a analizar simbolos, hay {len(simbolos_spot)} simbolos a analizar")

    for i, simbolo in tqdm.tqdm(enumerate(simbolos_spot)):
        # Agregar resultado al DataFrame (solo algunos para no saturar)
        if i % update_interval == 0:
            progress_bar.progress(i / len(simbolos_spot))
        try:
            # Preparar el dataset
            df_mayor = preparar_dataset(cliente, simbolo, interval, fecha_inicio, use_binance=1)
            # Analizador de velas
            analVelas = AnalizadorVelas(df_mayor, rango_analisis=rango_analisis)
            
            if dict_parametros:
                analIndi = AnalizadorIndicadores(df_mayor, dict_parametros)
                analIndi.Genera_Indicadores()
            #if "Media_Movil" in dict_parametros.keys() and "Estocastico" in dict_parametros.keys():              
             #   dict_estrategia1 = analIndi.Estrategia_MM_Estocastico()
                
            # Analizar las velas
            dict_grl = analVelas.analizar()

            # Agregar el resultado de std max-min
            dict_std = analVelas._std_max_min()
            dict_grl["Std_Min_Max"] = dict_std
            
            # Ultimas 3 velas
            ultimos_3_velas = [
                dict_grl["Velas"][k]["Vela_Ind"]
                for k in sorted(dict_grl["Velas"].keys(), key=lambda x: pd.to_datetime(x))[-5:]
            ]

            # Conteo de F
            # Aplanar la estructura para que sea compatible con DataFrame
            #print(dict_grl.keys())
            #print("**** ",dict_grl["Resumen_Swings"])
            fila = {
                "Simbolo": simbolo,
                "N": df_mayor.shape[0],
                "Volumen_acum": df_mayor["Volumne"].sum(),
                "Volumen": df_mayor["Volumne"].values[-1],
                "Tipo_Vela": ultimos_3_velas,
                "Cantidad_FVGs": len(dict_grl["FVGs"].keys()),
                "N_FVGs_Falta":dict_grl["FVGs_no_tocadas"]["total_no_tocadas"] ,
                "%_Cumplimiento_FVG": round( (dict_grl["FVGs_no_tocadas"]["pct_tocadas"]),2),
                "Ultimo_FVG_Falta": dict_grl["FVGs_no_tocadas"]['ultimas_fvg_info'],
                "Rupt_Mechas_Alcistas": dict_grl["Conteo_Rupturas"]["Mechas alcistas"],
                "Rupt_Mechas_Bajistas": dict_grl["Conteo_Rupturas"]["Mechas bajistas"],
                "Rupt_Velas_Alcistas": dict_grl["Conteo_Rupturas"]["Velas alcistas"],
                "Rupt_Velas_Bajistas": dict_grl["Conteo_Rupturas"]["Velas bajistas"],
                "Iteraciones": dict_grl["Conteo_Rupturas"]["Iteraciones"],
                "Min_Low": dict_grl["Rango_Min_Max"][0],
                "Max_High": dict_grl["Rango_Min_Max"][1],
                "Precio_Actual": dict_std["Precio_Actual"],
                "Porcentaje_Alcista": dict_std["Porcentaje_Alcista"],
                "Porcentaje_Bajista": dict_std["Porcentaje_Bajista"], 
                "Diferencia_Porcentajes": dict_std["Diferencia_Porcentajes"],
                "Total_Swing_Alcistas": dict_grl["Resumen_Swings"].get("Total_Swing_Alcistas"),
                "Total_Swing_Bajistas": dict_grl["Resumen_Swings"].get("Total_Swing_Bajistas"),
                "Fecha_Ultimo_Swing_Alcista": dict_grl["Resumen_Swings"].get("Fecha_Ultimo_Swing_Alcista"),
                "Precio_Ultimo_Swing_Alcista": dict_grl["Resumen_Swings"].get("Precio_Ultimo_Swing_Alcista"),
                "Fecha_Ultimo_Swing_Bajista": dict_grl["Resumen_Swings"].get("Fecha_Ultimo_Swing_Bajista"),
                "Precio_Ultimo_Swing_Bajista": dict_grl["Resumen_Swings"].get("Precio_Ultimo_Swing_Bajista"),
            }
            
            if dict_parametros:
                for _, dict_ind in analIndi.ult_value_indicator.items():
                    fila.update(dict_ind)

            resultados.append(fila)
            #print("paso el dict")
        
        except Exception as e:
            error_trace = traceback.format_exc()  # Obtiene toda la traza del error
            st.error(f"❌ Error al durante el analisis de simbolos para {simbolo} en intervalo {interval}:\n{error_trace}")
            continue
    progress_bar.progress(1.0)

    # Convertir a DataFrame final
    df_resultado = pd.DataFrame(resultados)
    st.write("Termino de analizar simbolos")
    return df_resultado

def filtrar_dataframe(df):
    st.subheader("🔍 Filtrado Flexible de Datos")

    # 1️⃣ Selección de columnas (toda la página)
    columnas = st.multiselect(
        "Selecciona las columnas a filtrar",
        options=df.columns.tolist()
    )

    # Inicializar máscara de todos True
    filtro_general = pd.Series(True, index=df.index)

    # Guardaremos las preferencias de ordenamiento en una lista de tuplas (col, ascendente)
    ordenamientos = []

    # 2️⃣ Para cada columna seleccionada, fila horizontal con filtros y orden
    for col in columnas:
        st.markdown(f"---\n### Filtro y orden para columna: **{col}**")

        # Crear 4 columnas: Filtro/Tipo | Valores | (opcional) Segundo valor | Ordenamiento
        c1, c2, c3, c4 = st.columns([1.2, 1.8, 2, 1.5])

        if pd.api.types.is_numeric_dtype(df[col]):
            # NUMÉRICO
            with c1:
                tipo_filtro = st.radio(
                    f"Tipo filtro '{col}'",
                    ("Mayor que", "Menor que", "Entre"),
                    key=f"radio_{col}"
                )

            if tipo_filtro == "Mayor que":
                with c2:
                    val = st.number_input(
                        f"Valor mínimo",
                        value=float(df[col].min()),
                        key=f"num_min_{col}"
                    )
                filtro = df[col] > val

            elif tipo_filtro == "Menor que":
                with c2:
                    val = st.number_input(
                        f"Valor máximo",
                        value=float(df[col].max()),
                        key=f"num_max_{col}"
                    )
                filtro = df[col] < val

            elif tipo_filtro == "Entre":
                with c2:
                    vmin, vmax = st.slider(
                        f"Rango",
                        min_value=float(df[col].min()),
                        max_value=float(df[col].max()),
                        value=(float(df[col].min()), float(df[col].max())),
                        key=f"slider_{col}"
                    )
                filtro = df[col].between(vmin, vmax)

        else:
            # TEXTO
            with c1:
                st.write("Filtro de texto")
            with c2:
                substring = st.text_input(
                    f"Subcadena a buscar",
                    "",
                    key=f"text_{col}"
                )
            filtro = df[col].str.contains(substring, case=False, na=False)

        # Combinar este filtro
        filtro_general &= filtro

        # Ordenamiento
        with c4:
            ordenar = st.checkbox(
                f"Ordenar '{col}'",
                key=f"ordenar_{col}"
            )
            if ordenar:
                asc = st.radio(
                    "Dirección",
                    ("Ascendente", "Descendente"),
                    key=f"ascdesc_{col}"
                )
                ordenamientos.append( (col, asc=="Ascendente") )

    # 3️⃣ Aplicar filtro
    df_filtrado = df[filtro_general]

    # 4️⃣ Aplicar ordenamientos si hay
    if ordenamientos:
        cols_orden = [c for c,a in ordenamientos]
        asc_orden = [a for c,a in ordenamientos]
        df_filtrado = df_filtrado.sort_values(by=cols_orden, ascending=asc_orden)

    # 5️⃣ Mostrar resultado
    st.markdown("---")
    st.write("✅ Resultado del filtrado y ordenamiento:")
    #st.dataframe(df_filtrado)
    #st.write(f"🔢 Filas seleccionadas: {df_filtrado.shape[0]} de {df.shape[0]}")

    return df_filtrado

def seleccionar_columnas(df: pd.DataFrame) -> list:
    import streamlit as st

    st.subheader("🧾 Selección de columnas a visualizar")
    columnas_seleccionadas = st.multiselect(
        "Selecciona las columnas que deseas mostrar:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )
    return columnas_seleccionadas

def seleccionar_columnas(df: pd.DataFrame) -> list:
    import streamlit as st

    st.subheader("🧾 Selección de columnas a visualizar")
    columnas_seleccionadas = st.multiselect(
        "Selecciona las columnas que deseas mostrar:",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )
    return columnas_seleccionadas

def convertir_columnas_a_dict(df: pd.DataFrame) -> pd.DataFrame:
    import ast
    for col in df.select_dtypes(include='object').columns:
        try:
            if df[col].apply(lambda x: isinstance(ast.literal_eval(str(x)), dict)).all():
                df[col] = df[col].apply(lambda x: ast.literal_eval(str(x)))
        except (ValueError, SyntaxError):
            continue  # Si falla la conversión, se deja como está
    return df

def solicitar_filtros_orden(df: pd.DataFrame):
    
    df = convertir_columnas_a_dict(df)
    st.subheader("🔍 Configuración de filtros y ordenamientos")

    columnas = st.multiselect(
        "Selecciona las columnas a aplicar filtros u orden",
        options=df.columns.tolist()
    )

    filtros = []
    ordenamientos = []

    for col in columnas:
        st.markdown(f"---\n### Filtro y orden para columna: **{col}**")
        c1, c2, c3, c4 = st.columns([1.2, 2, 2, 1.5])

        with c1:
            tipo_dato = "dict" if df[col].apply(lambda x: isinstance(x, dict)).any() else df[col].dtype

        if tipo_dato == "dict":
            claves_dict = set()
            df[col].dropna().apply(lambda x: claves_dict.update(x.keys()) if isinstance(x, dict) else None)
            clave_seleccionada = c2.selectbox(f"🔑 Clave del diccionario en {col}", list(claves_dict), key=f"clave_{col}")
            operacion = c3.selectbox(f"Condición", ["==", "!=", ">", "<", "in"], key=f"op_{col}")

            valor = c4.text_input(f"Valor a comparar", key=f"val_{col}")
            filtros.append((col, clave_seleccionada, operacion, valor))

        elif pd.api.types.is_numeric_dtype(df[col]):
            tipo_filtro = c1.radio(
                f"Tipo de filtro",
                ("Mayor que", "Menor que", "Entre"),
                key=f"radio_{col}"
            )

            if tipo_filtro == "Mayor que":
                val = c2.number_input("Valor mínimo", value=float(df[col].min()), key=f"min_{col}")
                filtros.append((col, ">", val))

            elif tipo_filtro == "Menor que":
                val = c2.number_input("Valor máximo", value=float(df[col].max()), key=f"max_{col}")
                filtros.append((col, "<", val))

            elif tipo_filtro == "Entre":
                vmin, vmax = c2.slider(
                    "Rango",
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=(float(df[col].min()), float(df[col].max())),
                    key=f"range_{col}"
                )
                filtros.append((col, "between", (vmin, vmax)))

        else:
            substring = c2.text_input(f"Subcadena a buscar en {col}", "", key=f"text_{col}")
            filtros.append((col, "contains", substring))

        with c4:
            ordenar = st.checkbox("Ordenar", key=f"ord_{col}")
            if ordenar:
                asc = st.radio("Orden", ["Ascendente", "Descendente"], key=f"asc_{col}")
                ordenamientos.append((col, asc == "Ascendente"))

    return filtros, ordenamientos

def aplicar_filtros_orden(df: pd.DataFrame, filtros, ordenamientos):
    df_filtrado = df.copy()
    #st.write(df_filtrado.columns)
    for filtro in filtros:
        if len(filtro) == 4:
            col, clave, op, val = filtro
            if op == "==":
                df_filtrado = df_filtrado[df_filtrado[col].apply(lambda x: x.get(clave) == val if isinstance(x, dict) else False)]
            elif op == "!=":
                df_filtrado = df_filtrado[df_filtrado[col].apply(lambda x: x.get(clave) != val if isinstance(x, dict) else False)]
            elif op == ">":
                df_filtrado = df_filtrado[df_filtrado[col].apply(lambda x: float(x.get(clave)) > float(val) if isinstance(x, dict) else False)]
            elif op == "<":
                df_filtrado = df_filtrado[df_filtrado[col].apply(lambda x: float(x.get(clave)) < float(val) if isinstance(x, dict) else False)]
            elif op == "in":
                df_filtrado = df_filtrado[df_filtrado[col].apply(lambda x: val in x.get(clave, "") if isinstance(x, dict) else False)]

        elif len(filtro) == 3:
            col, op, val = filtro
            if op == ">":
                df_filtrado = df_filtrado[df_filtrado[col] > val]
            elif op == "<":
                df_filtrado = df_filtrado[df_filtrado[col] < val]
            elif op == "between":
                df_filtrado = df_filtrado[df_filtrado[col].between(val[0], val[1])]
            elif op == "contains":
                df_filtrado = df_filtrado[df_filtrado[col].astype(str).str.contains(val, case=False, na=False)]

    if ordenamientos:
        cols_orden, asc_orden = zip(*ordenamientos)
        df_filtrado = df_filtrado.sort_values(by=cols_orden, ascending=asc_orden)

    return df_filtrado



def Poner_Indicadores(df, parametros_indicadores):
    cols_indicadores = {}
    if "Media_Movil" in parametros_indicadores.keys():
        m = Medias_Moviles()
        tipo_corta, tipo_larga, periodo_corta, periodo_larga = parametros_indicadores["Media_Movil"]
        df = m.ColocarMM(
            df,
            columna_precio="Close",
            tipo_corta=tipo_corta,
            periodo_corta=periodo_corta,
            tipo_larga=tipo_larga,
            periodo_larga=periodo_larga
        )
        
        cols_indicadores["Media_Movil"] = [f"{tipo_corta}_{periodo_corta}", f"{tipo_larga}_{periodo_larga}"]
        
    return df, cols_indicadores

def obtener_parametros_globales_MF():
    st.title("📊 Parámetros de Análisis Multi-Timeframe")

    timeframes = st.multiselect(
        "Selecciona los intervalos",
        ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1W", "1M"],
        default=["4h", "1d"]
    )

    st.markdown("### Rango temporal por Timeframe")

    parametros_por_tf = {}

    for tf in timeframes:
        with st.expander(f"⚙️ Parámetros para `{tf}`", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                periodos = st.number_input(f"{tf} - Periodos a descargar", min_value=1, max_value=10000, value=90, key=f"desc_{tf}")
                periodicidad = st.selectbox(f"{tf} - Periodicidad", ["minutos", "horas", "días", "semanas"], index=2, key=f"per_{tf}")
            with col2:
                excluir_final = st.number_input(f"{tf} - Periodos a excluir", min_value=0, max_value=10000, value=7, key=f"excluir_{tf}")
                rango_analisis = st.number_input(f"{tf} - Periodos a analizar", min_value=1, max_value=10000, value=60, key=f"analizar_{tf}")

            delta_dict = {
                "minutos": timedelta(minutes=periodos),
                "horas": timedelta(hours=periodos),
                "días": timedelta(days=periodos),
                "semanas": timedelta(weeks=periodos)
            }
            fecha_inicio = datetime.now() - delta_dict[periodicidad]
            fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d %H:%M:%S')

            st.markdown(f"🕒 Fecha de inicio para `{tf}`: **{fecha_inicio_str}**")

            parametros_por_tf[tf] = (fecha_inicio_str, excluir_final, rango_analisis)

    return parametros_por_tf


def cargar_df_existente(intervalo):
    interval_suffix = intervalo.lower()
    path = f"{ruta_data}/Analisis_simbolos_{interval_suffix}.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        timestamp_mod = os.path.getmtime(path)
        fecha_mod = datetime.fromtimestamp(timestamp_mod).strftime("%Y-%m-%d %H:%M:%S")

        st.success(f"📁 Datos cargados de `{intervalo}`")
        st.markdown(f"🕒 Última modificación: `{fecha_mod}`")

        return df, path, fecha_mod
    else:
        st.warning(f"❌ No se encontró archivo previo para `{intervalo}`.")
        return None, path, None
    
def generar_df_nuevo(simbolo, intervalo, fecha_inicio, rango_analisis, dict_parametros, cliente):
    interval_suffix = intervalo.lower()
    path = f"{ruta_data}/Analisis_simbolos_{interval_suffix}.csv"
    try:
        df = analizar_simbolos([simbolo], cliente, intervalo, fecha_inicio, rango_analisis=rango_analisis, dict_parametros=dict_parametros)
        if not df.empty:
            df.to_csv(path, index=False)
            st.success(f"✅ Archivo guardado como: `{path}`")
        else:
            st.warning(f"⚠️ Análisis para `{intervalo}` no arrojó resultados.")
        return df
    except Exception as e:
        error_trace = traceback.format_exc()  # Obtiene toda la traza del error
        st.error(f"❌ Error al generar nuevo df para {simbolo} en intervalo {intervalo  }:\n{error_trace}")
        return pd.DataFrame()
    
def forzar_actualizacion_dataframes(simbolos, dict_grl_frame, cliente, dict_parametros):    
    st.markdown("### 🔁 Actualización manual de archivos existentes")
    if st.button("🔃 Forzar actualización de todos los archivos"):
        with st.spinner("Actualizando archivos..."):
            actualizados = []
            for tf, (fecha_inicio, excluir_final, rango_analisis) in dict_grl_frame.items():
                df = analizar_simbolos(
                    simbolos,  # Limita la muestra
                    cliente, 
                    tf, 
                    fecha_inicio, 
                    rango_analisis=(excluir_final, rango_analisis), 
                    dict_parametros=dict_parametros
                )
                if df is not df.empty:
                    # Guardar con sufijo y fecha
                    interval_suffix = tf.lower() if isinstance(tf, str) else "unknown"
                    nombre_archivo_csv = f"{ruta_data}/Analisis_simbolos_{interval_suffix}.csv"
                    fecha_actual = datetime.now().strftime("%Y%m%d_%H%M")
                    nombre_guardado = f"analisis_simbolos_{interval_suffix}_{fecha_actual}.csv"
                    df.to_csv(nombre_archivo_csv, index=False)
                    print(f"✅ Análisis completado. Archivo guardado como: `{nombre_guardado}`")
                    st.success(f"✅ Análisis completado. Archivo guardado como: `{nombre_guardado}`")
            



def ordenar_columnas(cols):
    # Siempre primero "Simbolo"
    ordenadas = ["Simbolo"]

    # Sacamos Simbolo para no duplicar
    otras = [c for c in cols if c != "Simbolo"]

    # Agrupamos por el prefijo (parte antes del "_")
    prefijos = {}
    for c in otras:
        if "_" in c:
            prefijo, sufijo = c.rsplit("_", 1)
            prefijos.setdefault(prefijo, []).append((sufijo, c))
        else:
            prefijos.setdefault(c, []).append(("", c))

    # Para cada prefijo, ponemos menor → mayor
    for prefijo, lista in prefijos.items():
        lista_ordenada = sorted(lista, key=lambda x: (x[0] != "menor", x[0]))
        ordenadas.extend([col for _, col in lista_ordenada])

    return ordenadas