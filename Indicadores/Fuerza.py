import pandas as pd

def calcular_rsi(datos, ventana=14):
    '''
    (Function)
        Esta funciona returna el RSI como una Serie 
    (Args)
        datos: [pd.Series] Columna con la cual se generara el RSI
        ventana: [int] Funciona como referencia para la cantidad de datos que se tomaran para el RSI
    (Example)
        df["RSI"] = calcular_rsi(df["Close"], ventana=ventanaRSI)
    
    '''
    # Calcular cambios en precios
    delta = datos.diff()

    # Eliminar primer elemento NaN
    delta = delta.dropna()

    # Separar cambios positivos y negativos
    ganancias = delta.where(delta > 0, 0)
    pérdidas = delta.where(delta < 0, 0)

    # Calcular promedios móviles
    promedio_ganancias = ganancias.rolling(window=ventana, min_periods=ventana).mean()
    promedio_pérdidas = pérdidas.rolling(window=ventana, min_periods=ventana).mean()

    # Calcular el RSI
    rs = promedio_ganancias / promedio_pérdidas.abs()
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calcula_tsv(datos, opcion=2):
    """
    Calcula el Oscilador de Volumen Segmentado en el Tiempo (TSV).

    Args:
        datos (DataFrame): DataFrame que contiene las columnas 'Close' y 'Volume'.
        opcion (int): Si es 1 el tsv es el volumen en base al movimiento pero si es 2 el tsv sera el volumen
                    multiplicado el cambio de precio, lo cual le da mas valor. La diferencia principal sera la grafica 
                    que se visualiza con picos mas grandes (para opcion=2)
    Returns:
        pandas.Series: Serie que contiene los valores del TSV.
    """
    # Calcula el cambio en el precio de cierre
    cambio_precio = datos['Close'].diff()
    
    if opcion==1:
        # Inicializa una serie para almacenar los valores del TSV
        tsv = pd.Series(index=datos.index)
        
        # Calcula el TSV
        for i in range(1, len(datos)):
            if cambio_precio[i] > 0:
                tsv[i] = datos['Volume'][i]
            elif cambio_precio[i] < 0:
                tsv[i] = -datos['Volume'][i]
            else:
                tsv[i] = 0
    elif opcion==2:
        # Calcula el TSV utilizando el cambio de precio y el volumen
        tsv = cambio_precio * datos['Volume']
        
    return tsv