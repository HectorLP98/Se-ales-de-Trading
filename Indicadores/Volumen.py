import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcular_obv(precios, volumenes):
    """
    Calcula el On-Balance Volume (OBV) basado en un array de precios y un array de volúmenes.
    
    Args:
        precios (array-like): Array de precios de cierre.
        volumenes (array-like): Array de volúmenes correspondientes a los precios.

    Returns:
        numpy.ndarray: Array con los valores del OBV.
    """
    # Asegurarse de que los inputs sean arrays de numpy
    precios = np.array(precios)
    volumenes = np.array(volumenes)
    
    # Inicializar el array del OBV con el primer valor en 0
    obv = [0]
    
    # Calcular el OBV
    for i in range(1, len(precios)):
        if precios[i] > precios[i - 1]:
            obv.append(obv[-1] + volumenes[i])
        elif precios[i] < precios[i - 1]:
            obv.append(obv[-1] - volumenes[i])
        else:  # Si los precios son iguales
            obv.append(obv[-1])
    
    return np.array(obv)

def generate_plot_freq_time(df,ax, col_use="ct", tipo="hour",simbolo="None"):
    """_summary_
    Esta funcion plotea la suma del volumen 

    Args:
        df (_type_): _description_
        col_use (str, optional): Columna que se usara para determinar el tiempo. Defaults to "ct" que es Close_Time
        tipo (str, optional): hour o day, para el obtener el tipo de frecuencia. Defaults to "hour".

    Returns:
        ploty: figura con el grafico en ploty
    """
    
    if tipo.lower() == "hour":
        col_aux = "hour"
        title = f'Frecuencia Total de volumne {simbolo} por hora '
        df[col_aux] = df[col_use].dt.hour
        ordered_days = [i for i in range(0,25)]
    elif tipo.lower() == "day":
        col_aux = "day_name" 
        title = f'Frecuencia Total de volumne {simbolo} por dia'
        df[col_aux] = df[col_use].dt.day_name()
        # Definir el orden de los días de la semana de lunes a domingo
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    
    df_freq = df.groupby(col_aux, as_index=False).agg({"Volumne":sum})
    
    

    # Convertir la columna 'day_name' a un tipo categórico con el orden definido
    df_freq[col_aux] = pd.Categorical(df_freq[col_aux], categories=ordered_days, ordered=True)

    # Ordenar el DataFrame según la columna 'day_name'
    df_freq.sort_values(col_aux, inplace=True)
    df_freq.reset_index(drop=True, inplace=True)

    # Crear la gráfica de barras
    #fig = go.Figure(data=[go.Bar(x=df_freq[col_aux], y=df_freq.Volumne)])
    # Crear la gráfica de barras
    #plt.figure(figsize=(10, 6))
    ax.bar(df_freq[col_aux], df_freq["Volumne"], color='skyblue')
    # Configurar el título y las etiquetas
    ax.set_title(title)
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Volumen transaccionado')
    
    
def calculate_vpvr(prices, volumes, num_bins=20, calc_type='total'):
    """
    Calcula el perfil de volumen visible (VPVR) para un conjunto de precios y volúmenes.
    Permite calcular volumen total, delta de volumen o volumen up/down.

    Args:
        prices (list or np.array): Lista de precios.
        volumes (list or np.array): Lista de volúmenes correspondientes a los precios.
        num_bins (int): Número de intervalos (bins) para dividir el rango de precios. Por defecto 20.
        calc_type (str): Tipo de cálculo a realizar. Puede ser 'total', 'delta' o 'updown'.
                         - 'total': Volumen total en cada nivel de precio.
                         - 'delta': Diferencia entre volumen de compras y ventas.
                         - 'updown': Volumen up/down basado en cambios de precio.

    Returns:
        bin_edges (np.array): Bordes de los intervalos de precios.
        volume_profile (np.array): Volumen total (o delta/up-down) en cada intervalo de precios.
        poc_price (float): El precio correspondiente al POC (Point of Control).
    """
    # Definir los bordes de los intervalos de precios (bins) en función del rango de precios
    price_min, price_max = np.min(prices), np.max(prices)
    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    
    # Inicializar un array para almacenar el volumen en cada bin
    volume_profile = np.zeros(num_bins)

    # Alinear los precios con np.roll para evitar diferencias de tamaño
    prev_prices = np.roll(prices, 1)

    # Para cada bin, sumar el volumen de las transacciones que ocurrieron dentro del rango de precios del bin
    for i in range(num_bins):
        in_bin = (prices >= bin_edges[i]) & (prices < bin_edges[i + 1])
        
        if calc_type == 'total':
            # Calcular volumen total en cada bin
            volume_profile[i] = np.sum(volumes[in_bin])
        
        elif calc_type == 'delta':
            # Calcular volumen de compras (precios suben) y ventas (precios bajan) en cada bin
            up_volume = np.sum(volumes[in_bin & (prices > prev_prices)])
            down_volume = np.sum(volumes[in_bin & (prices < prev_prices)])
            # Delta es la diferencia entre compras y ventas
            volume_profile[i] = up_volume - down_volume
        
        elif calc_type == 'updown':
            # Calcular volumen de compras (precios suben) y ventas (precios bajan) en cada bin
            up_volume = np.sum(volumes[in_bin & (prices > prev_prices)])
            down_volume = np.sum(volumes[in_bin & (prices < prev_prices)])
            # Guardar el volumen de compras y ventas por separado
            volume_profile[i] = up_volume if up_volume > down_volume else -down_volume

    # Calcular el POC (precio con el volumen más alto)
    max_volume_index = np.argmax(volume_profile)
    poc_price = (bin_edges[max_volume_index] + bin_edges[max_volume_index + 1]) / 2

    return bin_edges, volume_profile, poc_price

# Calculate decreasing volume
def detect_decreasing(serie):
    """Detect if the volume oscillator is decreasing."""
    decreesing = serie.diff().lt(0)
    return decreesing


def simple_volume_oscillator(volume, period):
    """
    Calcula el Oscilador de Volumen Simple (SVO).

    :param volume: Serie de tiempo del volumen.
    :param period: Número de períodos para la SMA del volumen.
    :return: Serie de tiempo del SVO.
    """
    # Calcula la Media Móvil Simple (SMA) del volumen
    sma_volume = volume.rolling(window=period).mean()
    # Calcula el Oscilador de Volumen Simple (SVO)
    svo = (volume - sma_volume) / sma_volume
    return svo

# Function to calculate Volume Oscillator manually
def exponencial_volume_oscillator(volume, period):
    """Calculate the Volume Oscillator (VO) without talib."""
    ema_volume = volume.ewm(span=period, adjust=False).mean()
    evo = (volume - ema_volume) / ema_volume
    return evo
'''
# Función para el Índice de Flujo de Dinero (MFI)
def calculate_mfi(df, n):
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['MFV'] = df['TP'] * df['Volume']
    df['Positive_MFV'] = np.where(df['TP'] > df['TP'].shift(1), df['MFV'], 0)
    df['Negative_MFV'] = np.where(df['TP'] < df['TP'].shift(1), df['MFV'], 0)
    positive_mfv_sum = df['Positive_MFV'].rolling(window=n).sum()
    negative_mfv_sum = df['Negative_MFV'].rolling(window=n).sum()
    mfi = 100 - (100 / (1 + (positive_mfv_sum / negative_mfv_sum)))
    return mfi
'''

def calculate_mfi(high, low, close, volume, n):
    """
    Calcula el Índice de Flujo de Dinero (MFI).
    
    :param high: Serie de altos.
    :param low: Serie de bajos.
    :param close: Serie de cierres.
    :param volume: Serie de volúmenes.
    :param n: Periodo para el cálculo del MFI.
    :return: Serie de tiempo del MFI.
    """
    tp = (high + low + close) / 3
    mfv = tp * volume
    positive_mfv = np.where(tp > np.roll(tp, 1), mfv, 0)
    negative_mfv = np.where(tp < np.roll(tp, 1), mfv, 0)
    positive_mfv_sum = pd.Series(positive_mfv).rolling(window=n).sum()
    negative_mfv_sum = pd.Series(negative_mfv).rolling(window=n).sum()
    mfi = 100 - (100 / (1 + (positive_mfv_sum / negative_mfv_sum)))
    return mfi

# Calcular el volumen total por nivel de precio
def Zonas_Oferta_Demanda(df1, columns_use={"precio":"Close","volumen":"Volumne","tiempo":"Close_Time"},n_candels=100):
    df = df1.copy()
    df_aux = df.tail(n_candels)
    volume_by_price = df_aux.groupby(df_aux[columns_use["precio"]])[columns_use["volumen"]].sum()

    # Identificar los niveles de precios con el mayor volumen
    high_volume_levels = volume_by_price.sort_values(ascending=False).head(4)
    
    # Definir zonas de oferta y demanda (basadas en análisis de volumen)
    demand_zone = high_volume_levels.sort_index().index[0:2]
    supply_zone = high_volume_levels.sort_index().index[2:]
    print(f"Precios Demanda: {demand_zone}")
    print(f"Precios Oferta:  {supply_zone}")
    #demand_zone = (high_volume_levels.index.min(), np.percentile(high_volume_levels.index,20))
    #supply_zone = (np.percentile(high_volume_levels.index,80), high_volume_levels.index.max())

    # Visualizar el gráfico
    plt.figure(figsize=(14, 7))

    # Graficar el precio de cierre
    plt.plot(df[columns_use["tiempo"]], df[columns_use["precio"]], label='Precio de Cierre', color='black')

    # Colorear la zona de demanda
    plt.fill_between(df[columns_use["tiempo"]].tail(n_candels), demand_zone[0], demand_zone[1], color='green', alpha=0.3, label='Zona de Demanda')

    # Colorear la zona de oferta
    plt.fill_between(df[columns_use["tiempo"]].tail(n_candels), supply_zone[0], supply_zone[1], color='red', alpha=0.3, label='Zona de Oferta')

    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.title('Precio de Cierre con Zonas de Oferta y Demanda')
    plt.legend()
    plt.show()
