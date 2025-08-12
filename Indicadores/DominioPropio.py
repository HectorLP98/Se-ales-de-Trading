import pandas as pd
import numpy as np

def contar_cruces_ema(close_series, ema_series, window=20):
    cruces = 0
    for i in range(1, window):
        if (close_series[-i] > ema_series[-i] and close_series[-i-1] < ema_series[-i-1]) or \
           (close_series[-i] < ema_series[-i] and close_series[-i-1] > ema_series[-i-1]):
            cruces +=1
    return cruces

def distancia_ema_sma(ema_last, sma_last):
    return abs(ema_last - sma_last) / sma_last * 100  # %


def duracion_lateral(high_series, low_series, threshold=0.2):
    count = 0
    for h,l in zip(reversed(high_series), reversed(low_series)):
        rango = (h - l)/l*100
        if rango < threshold:
            count +=1
        else:
            break
    return count

def indice_calidad_pullback(df, window_cruces=20, threshold_rango=0.2, columnas_medias=[]):
    col_ema, col_sma = columnas_medias
    cruces = contar_cruces_ema(df["Close"], df[col_ema], window=window_cruces)
    distancia = distancia_ema_sma(df[col_ema].iloc[-1], df[col_sma].iloc[-1])
    lateral = duracion_lateral(df["High"], df["Low"], threshold=threshold_rango)
    
    # Normalizar para puntaje final (ajusta pesos según backtest)
    # Area de oportunidad para mejorar dinamicamente estos pesos. Con modelo
    score = (
        (5 - min(cruces,5)) * 0.4 +
        min(distancia,5) * 0.4 +
        (5 - min(lateral,5)) * 0.2
    )
    
    return score

def calcular_cruce_y_duracion(self, df, col_corta, col_larga):
    """
    Calcula el cruce y la duración del trend.
    """
    cruces = np.zeros(len(df), dtype=int)
    duracion = np.zeros(len(df), dtype=int)

    corta = df[col_corta].values
    larga = df[col_larga].values

    for i in range(1, len(df)):
        if np.isnan(corta[i]) or np.isnan(larga[i]) or np.isnan(corta[i - 1]) or np.isnan(larga[i - 1]):
            duracion[i] = duracion[i - 1]
            continue

        # Detectar cruce
        if corta[i] > larga[i] and corta[i - 1] <= larga[i - 1]:
            cruces[i] = 1
            duracion[i] = 0
        elif corta[i] < larga[i] and corta[i - 1] >= larga[i - 1]:
            cruces[i] = -1
            duracion[i] = 0
        else:
            duracion[i] = duracion[i - 1] + 1

    return cruces, duracion

def obtener_min_max(df, rango_analisis):
    """
    Devuelve el mínimo del campo 'Low' y el máximo del campo 'High',
    junto con sus índices, usando un rango definido por una tupla.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'Low' y 'High'.
        rango_analisis (tuple): (excluir_ultimas, tomar_n)
            excluir_ultimas: número de filas a excluir desde el final.
            tomar_n: número de filas a tomar desde el final después de la exclusión.

    Returns:
        tuple: (min_low, max_high, idx_min, idx_max)
    """
    excluir_ultimas, tomar_n = rango_analisis

    if len(df) <= excluir_ultimas:
        raise ValueError("El número de filas a excluir es mayor o igual al tamaño del DataFrame.")

    df_filtrado = df.iloc[:-excluir_ultimas] if excluir_ultimas > 0 else df.copy()
    df_filtrado = df_filtrado.iloc[-tomar_n:] if tomar_n > 0 else df_filtrado

    min_low = df_filtrado["Low"].min()
    max_high = df_filtrado["High"].max()

    idx_min = df_filtrado["Low"].idxmin()
    idx_max = df_filtrado["High"].idxmax()

    return min_low, max_high, idx_min, idx_max

def detectar_swing_points(df_original):
    """
    Detecta los puntos de swing (cambios de dirección locales) en el precio.
    Agrega dos nuevas columnas al DataFrame original:
    - Swing_Alcista: 1 si es un mínimo local (Low)
    - Swing_Bajista: 1 si es un máximo local (High)
    """
    df = df_original.copy()
    df["Swing_Alcista"] = 0
    df["Swing_Bajista"] = 0

    for i in range(1, len(df) - 1):
        # Swing Alcista (mínimo local)
        if df["Low"].iloc[i] < df["Low"].iloc[i - 1] and df["Low"].iloc[i] < df["Low"].iloc[i + 1]:
            df.at[df.index[i], "Swing_Alcista"] = 1

        # Swing Bajista (máximo local)
        if df["High"].iloc[i] > df["High"].iloc[i - 1] and df["High"].iloc[i] > df["High"].iloc[i + 1]:
            df.at[df.index[i], "Swing_Bajista"] = 1
    return df

def Contar_FVGs(df,dict_fvg):
    fvg_no_tocadas = {k: v for k, v in dict_fvg.items() if not v.get("tocada", True)}
    # Ponemos los indices
    indices = []
    for k, v in fvg_no_tocadas.items():
        indices.append(pd.to_datetime(str(v['fecha_fin'])))
    df["FVGs_Falta"] = 0
    df.loc[indices,"FVGs_Falta"] = 1
    df["Duracion_FVGs_Falta"] = (
                                    df.groupby((df["FVGs_Falta"] == 1).cumsum()).cumcount() + 1
                                )
    return df

def Color_Vela(open_, close):
    if close > open_:
        return 1
    elif close < open_:
        return -1
    else:
        return 0

def Tipo_Vela(row=None, open_=None, high=None, low=None, close=None, pt1=0.5, pt2=0.05, pt3=0.4):
    '''
    (Function)
        funcion clasifica la vela en funcion de la mechas y el cuerpo de la vela
    (Parameters)
        - row: diccionario o Serie de pandas (con claves: Open, Close, High, Low)
        - open_, high, low, close: valores individuales (si no se pasa row)
        . pt1: Porcion de la mecha mayoritaria (la parte mas grande de una mecha en proporcion al total de la vela) default 0.5, es decir la mecha debe medir al menos la mitad del total de la vela
        - pt2: Porcion del mecha menoritaria o tolerancia. Default 0.05, es decir la mecha minoritaria debe ser a lo mas el 5% del total de la vela.
        - pt3: Porcion de las mechas para considerarse Doji, se parte de una simetria con limite superior. Default .4, quiere decir que el cuerpo de la vela no supera el 20% para ser considerada Doji
    (Return)
        str: "Martillo", "Martillo_Inv", "Hang_Man", "Estrella_Fug", "Doji", "Cuerpo_Lleno"
    '''
    # Obtener datos
    if row is not None:
        open_ = row["Open"]
        high = row["High"]
        low = row["Low"]
        close = row["Close"]
        
    # Validación
    if any(v is None for v in [open_, high, low, close]):
        raise ValueError( f"Error al pasar paremtros en Tipo_Vela")
    
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
    else:
        return "N/A"