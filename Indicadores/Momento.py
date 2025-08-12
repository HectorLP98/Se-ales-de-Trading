import pandas as pd
import numpy as np

def calcular_rsi(series_precios, periodos=14):
    """
    Calcula el Índice de Fuerza Relativa (RSI) para una serie de precios.

    Args:
        series_precios (pd.Series): Serie de precios del activo.
        periodos (int): Número de periodos para calcular el RSI. El valor por defecto es 14.

    Returns:
        pd.Series: Serie con el valor del RSI para cada punto temporal.
    """
    # Calcular diferencias entre precios consecutivos
    diferencia = series_precios.diff(1)
    
    # Separar ganancias y pérdidas
    ganancias = diferencia.where(diferencia > 0, 0)
    perdidas = -diferencia.where(diferencia < 0, 0)
    
    # Calcular las medias móviles de las ganancias y pérdidas
    ganancia_promedio = ganancias.rolling(window=periodos, min_periods=1).mean()
    perdida_promedio = perdidas.rolling(window=periodos, min_periods=1).mean()
    
    # Calcular el RS y el RSI
    rs = ganancia_promedio / perdida_promedio
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def generar_senal_rsi(arr_rsi, sobreventa=30, sobrecompra=70):
    """
    Genera un vector con las señales RSI dadas las condiciones de sobrecompra y sobreventa.
    
    Parámetros:
    - arr_rsi: np.ndarray o pd.Series con los valores de RSI.
    - sobreventa: umbral inferior (default=30).
    - sobrecompra: umbral superior (default=70).
    
    Return:
    - np.ndarray con valores: "Sobre-venta", "Sobre-compra", "Neutro"
    """
    arr_rsi = np.asarray(arr_rsi)
    condiciones = [
        arr_rsi < sobreventa,
        arr_rsi > sobrecompra
    ]
    elecciones = ['Sobre-venta', 'Sobre-compra']
    return np.select(condiciones, elecciones, default='Neutro')


def calcular_duracion_senal(senales):
    """
    Calcula la duración consecutiva de cada tipo de señal en una serie.
    
    Parámetros:
    - senales: np.ndarray o pd.Series con valores categóricos como "Sobre-venta", "Neutro", etc.
    
    Return:
    - np.ndarray con la duración consecutiva de la señal actual.
    """
    senales = np.asarray(senales)
    duraciones = np.zeros(len(senales), dtype=int)
    contador = 1

    for i in range(1, len(senales)):
        if senales[i] == senales[i - 1]:
            contador += 1
        else:
            contador = 1
        duraciones[i] = contador

    return duraciones


def calcular_estocastico(df, n=14, d=3):
    """
    Calcula el Indicador Estocástico.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas 'High', 'Low' y 'Close'.
        n (int): Número de periodos para calcular el rango alto-bajo (por defecto 14).
        d (int): Número de periodos para calcular la media móvil de %K (por defecto 3).
        
    Returns:
        pd.DataFrame: DataFrame con las columnas %K y %D añadidas.
    """
    
    # Calcular el máximo y mínimo de los últimos 'n' periodos
    df['L_n'] = df['Low'].rolling(window=n).min()  # Mínimo de los últimos n períodos
    df['H_n'] = df['High'].rolling(window=n).max()  # Máximo de los últimos n períodos
    
    # Calcular %K
    df['%K'] = 100 * ((df['Close'] - df['L_n']) / (df['H_n'] - df['L_n']))
    
    # Calcular %D (media móvil simple de %K)
    df['%D'] = df['%K'].rolling(window=d).mean()
    
    # Eliminar las columnas temporales L_n y H_n
    df = df.drop(columns=['L_n', 'H_n'])
    
    return df

# Generar señal combinada
def estocastico_signal(row):
    k, d = row['%K'], row['%D']
    if k > d:
        tendencia = "Alcista"
    else:
        tendencia = "Bajista"
    if k > 80 and d > 80:
        sobre = "Sobre-compra"
    elif k < 20 and d < 20:
        sobre = "Sobre-venta"
    elif k > 80:
        sobre = "Sobre-compra-K"
    elif d > 80:
        sobre = "Sobre-compra-D"
    elif k < 20:
        sobre = "Sobre-venta-K"
    elif d < 20:
        sobre = "Sobre-venta-D"
    else:
        sobre = "Neutro"
    return f"{sobre}"

def calcular_duracion_estocastico_numpy(estados):
    """
    Calcula con NumPy la duración continua de cada estado del Estocástico.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene la columna de estado.
        col_estado (str): Nombre de la columna con la señal de estado.
    
    Returns:
        pd.DataFrame: Mismo DataFrame con columna 'Duracion_Estocastico'.
    """
    estados = np.array(estados)
    # Creamos array para duración
    duracion = np.zeros_like(estados, dtype=float)
    
    contador = 0
    estado_anterior = None

    for i in range(len(estados)):
        estado_actual = estados[i]
        
        if pd.isna(estado_actual):
            duracion[i] = np.nan
            contador = 0
            estado_anterior = None
            continue

        if estado_actual == estado_anterior:
            contador += 1
        else:
            contador = 1
        
        duracion[i] = contador
        estado_anterior = estado_actual

    return duracion