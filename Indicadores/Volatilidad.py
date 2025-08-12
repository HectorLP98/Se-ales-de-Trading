import pandas as pd

def calcular_bandas_bollinger(df, columna_precio='Close', ventana=20, num_std=2):
    """
    Calcula las Bandas de Bollinger para una serie de precios.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos de mercado.
        columna_precio (str, optional): Nombre de la columna de precios. Defaults to 'Close'.
        ventana (int, optional): Número de períodos para calcular la media móvil. Defaults to 20.
        num_std (int, optional): Número de desviaciones estándar para las bandas. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame con las Bandas de Bollinger añadidas.
    """
    
    # Calcula la media móvil (Banda Media)
    df['Banda_Media'] = df[columna_precio].rolling(window=ventana).mean()
    
    # Calcula la desviación estándar
    df['Desviacion_Estandar'] = df[columna_precio].rolling(window=ventana).std()
    
    # Calcula las Bandas de Bollinger Superior e Inferior
    df['Banda_Superior'] = df['Banda_Media'] + (df['Desviacion_Estandar'] * num_std)
    df['Banda_Inferior'] = df['Banda_Media'] - (df['Desviacion_Estandar'] * num_std)
    
    return df

def calculate_atr(df1, n=14):
    """
    Calcula el Average True Range (ATR) de un DataFrame de precios.

    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas 'High', 'Low', 'Close'.
        n (int): El número de períodos para calcular el ATR (por defecto es 14).

    Returns:
        pd.Series: Una serie con los valores del ATR.
    """
    df = df1.copy()
    # Calcular True Range (TR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    
    # Calcular ATR como media móvil del True Range
    atr = df['TR'].rolling(window=n, min_periods=1).mean()
    
    return atr