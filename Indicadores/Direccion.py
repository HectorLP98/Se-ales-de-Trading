import pandas as pd
import numpy as np 


# Sobre el macd
def normalize_array(arr):
    # Asegurarse de que el array no esté vacío
    if arr.size == 0:
        return arr
    
    # Calcular el mínimo y el máximo del array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalizar el array
    normalized = (arr - min_val) / (max_val - min_val)

    return normalized*100

class Medias_Moviles:

    def __init__(self):
        pass

    def calcular_WMA(self, data, window):
        """Media móvil ponderada"""
        weights = np.arange(1, window + 1)
        return data.rolling(window).apply(
            lambda prices: np.dot(prices, weights) / weights.sum(),
            raw=True
        )

    def calcular_SMA(self, data, window):
        """Media móvil simple"""
        return data.rolling(window).mean()

    def calcular_EMA(self, data, window):
        """Media móvil exponencial optimizada"""
        alpha = 2 / (window + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data.iloc[0]
        for i in range(1, len(data)):
            ema[i] = (data.iloc[i] - ema[i - 1]) * alpha + ema[i - 1]
        return pd.Series(ema, index=data.index)

    def obtener_media(self, data, tipo, window):
        """Obtiene la media móvil del tipo solicitado"""
        if tipo == "EMA":
            return self.calcular_EMA(data, window)
        elif tipo == "SMA":
            return self.calcular_SMA(data, window)
        elif tipo == "WMA":
            return self.calcular_WMA(data, window)
        else:
            raise ValueError(f"Tipo de media no soportado: {tipo}")

    def calcular_cruce_y_duracion(self, df, col_corta, col_larga):
        """
        Calcula el cruce y la duración del trend.
        """
        nombre_prefix = f"{col_corta}_{col_larga}"
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
                duracion[i] = 1
            elif corta[i] < larga[i] and corta[i - 1] >= larga[i - 1]:
                cruces[i] = -1
                duracion[i] = 1
            else:
                duracion[i] = duracion[i - 1] + 1

        df[f"CruceMM_{nombre_prefix}"] = cruces
        df[f"DuracionTrendMM_{nombre_prefix}"] = duracion

        return df

    def ColocarMM(self,df_input,columna_precio="Close",tipo_corta="EMA",periodo_corta=7,tipo_larga="SMA",periodo_larga=30):
        """
        Calcula dos medias móviles y añade cruce + duración.
        """
        df = df_input.copy()

        # Calcular medias móviles
        col_corta = f"{tipo_corta}_{periodo_corta}"
        col_larga = f"{tipo_larga}_{periodo_larga}"

        df[col_corta] = self.obtener_media(df[columna_precio], tipo_corta, periodo_corta)
        df[col_larga] = self.obtener_media(df[columna_precio], tipo_larga, periodo_larga)

        # Calcular cruce y duración
        df = self.calcular_cruce_y_duracion(df, col_corta, col_larga)

        return df

class Medias_Moviles1:
    def __init__(self):
        pass

    def generar_EMA(self, v1, n):
        v1 = pd.Series(v1)
        return v1.ewm(span=n, adjust=False).mean().to_numpy()

    def generar_SMA(self, v1, n):
        v1 = pd.Series(v1)
        return v1.rolling(window=n).mean().to_numpy()

    def generar_WMA(self, v1, n):
        weights = np.arange(1, n + 1)
        v1 = pd.Series(v1)
        return v1.rolling(window=n).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True).to_numpy()

    def generar_Cruce(self, col_corta, col_larga):
        col_corta = np.array(col_corta)
        col_larga = np.array(col_larga)

        cruce = np.zeros_like(col_corta, dtype=int)
        for i in range(1, len(col_corta)):
            if col_corta[i] > col_larga[i] and col_corta[i - 1] <= col_larga[i - 1]:
                cruce[i] = 1  # Cruce alcista
            elif col_corta[i] < col_larga[i] and col_corta[i - 1] >= col_larga[i - 1]:
                cruce[i] = -1  # Cruce bajista
        return cruce

    def calcular_medias_y_cruces(self, v1, ma_use, periodos_mm, prefijo="OBV_"):
        """
        Calcula medias móviles y cruces, y los guarda en un DataFrame.
        Args:
            v1 (list, tuple, np.ndarray): Serie de datos.
            ma_use (str): Tipo de media móvil ("EMA", "SMA", "WMA").
            periodos_mm (list): Lista de períodos [n1, n2, ..., nk].
            prefijo (str): Prefijo para los nombres de las columnas.

        Returns:
            pd.DataFrame: DataFrame con las medias móviles y cruces.
        """
        ma_use = ma_use.upper()
        if ma_use not in ["EMA", "SMA", "WMA"]:
            raise ValueError("El tipo de media móvil debe ser 'EMA', 'SMA' o 'WMA'.")

        resultados = {}

        # Ordenar los periodos y emparejar
        periodos_mm = sorted(periodos_mm)
        pares_periodos = [(n1, n2) for i, n1 in enumerate(periodos_mm) for n2 in periodos_mm[i + 1:]]

        for n1, n2 in pares_periodos:
            # Generar medias móviles según el tipo
            if ma_use == "EMA":
                ma_corta = self.generar_EMA(v1, n1)
                ma_larga = self.generar_EMA(v1, n2)
            elif ma_use == "SMA":
                ma_corta = self.generar_SMA(v1, n1)
                ma_larga = self.generar_SMA(v1, n2)
            elif ma_use == "WMA":
                ma_corta = self.generar_WMA(v1, n1)
                ma_larga = self.generar_WMA(v1, n2)

            # Nombres de las columnas
            col_corta_name = f"{prefijo}{ma_use}_{n1}"
            col_larga_name = f"{prefijo}{ma_use}_{n2}"
            col_cruce_name = f"{prefijo}Cruce_{ma_use}_{n1}_{n2}"

            # Guardar medias
            resultados[col_corta_name] = ma_corta
            resultados[col_larga_name] = ma_larga

            # Generar y guardar cruces
            resultados[col_cruce_name] = self.generar_Cruce(ma_corta, ma_larga)

        # Convertir resultados en un DataFrame
        df_resultados = pd.DataFrame(resultados)
        return df_resultados


def calcular_MACD(df, col_precio="Close", fastLength=12, slowLength=26, signalLength=9):
    """
    Calcula el MACD y lo añade al DataFrame.
    """
    mm = Medias_Moviles()
    ema_fast = mm.calcular_EMA(df[col_precio], fastLength)
    ema_slow = mm.calcular_EMA(df[col_precio], slowLength)

    df["MACD"] = ema_fast - ema_slow
    df["Signal_Line"] = mm.calcular_EMA(df["MACD"], signalLength)
    df["Histograma_MACD"] = df["MACD"] - df["Signal_Line"]

    # También puedes agregar señales si lo deseas:
    df["MACD_Senal"] = df.apply(
        lambda x: "Alcista" if x["MACD"] > x["Signal_Line"] else "Bajista",
        axis=1
    )

    return df
