import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Indicadores.Direccion import Medias_Moviles
from Indicadores.Estrategias import Estrategias
from datetime import datetime, timedelta
mm = Medias_Moviles()


def MapCount(ColInd,tipo="conteo"):
    
    new = []
    contador = 0
    inicia_contador = False
    tendencia = ""
    for i in ColInd:
        if i > 0:
            inicia_contador = True 
            if i==1: tendencia = "sobre-venta"
            elif i == 2: tendencia="sobre-compra"
        if i < 0:
            inicia_contador = False
        if inicia_contador:
            if tipo.lower() == "conteo": contador += 1
            elif tipo.lower() == "tendencia": contador = ""
        else:
            contador = 0
        
            tendencia = ""
        if tipo=="tendencia":
            new.append(tendencia)
        elif tipo=="conteo":
            new.append(contador)
        
        
        
    return np.array(new)
            

def ConteoTrend(df_test1,col_use,rsi_values, dropCols=False):
    df_test1[f"{col_use}"].fillna(0,inplace=True)
    df_test1[f"{col_use}_ant"] = df_test1[col_use].shift(1)
    df_test1[f"{col_use}_ant"] = df_test1[f"{col_use}_ant"].fillna(0)
    df_test1[f"{col_use}_Ind"] = df_test1.apply(lambda row: 1 if (row[col_use]< rsi_values[0] and row[f"{col_use}_ant"]> rsi_values[0]) else -1 if (row[col_use]>rsi_values[0] and row[f"{col_use}_ant"]<=rsi_values[0]) else 2 if (row[col_use]>rsi_values[1] and row[f"{col_use}_ant"]<=rsi_values[1]) else -2 if (row[col_use]<rsi_values[1] and row[f"{col_use}_ant"]>=rsi_values[1]) else 0,
                                                    axis=1)

    df_test1[f"Trend_{col_use}"] = MapCount(df_test1[f"{col_use}_Ind"], tipo="tendencia")
    df_test1[f"CountTrend_{col_use}"] = MapCount(df_test1[f"{col_use}_Ind"], tipo="conteo")
    if dropCols:
        df_test1.drop(columns=[f"{col_use}_Ind",f"{col_use}_ant"], inplace=True)
        
    return df_test1




def calcular_fibonacci(precios, reversa=False, niveles_fibonacci=[0.236, 0.382, 0.5, 0.618, 0.786]):
    # Encuentra el máximo y el mínimo en la lista de precios
    max_precio = max(precios)
    min_precio = min(precios)
    
    # Calcula el rango de precios
    rango = max_precio - min_precio
    #print(min_precio, max_precio)
    #print(rango)


    niveles = {}
    
    # Si reversa es True, calcular el Fibonacci al revés
    if reversa:
        # Fibonacci invertido: comenzando desde el mínimo y sumando el rango
        for nivel in niveles_fibonacci:
            niveles[f'Nivel {round((nivel)*100,2)}%'] = max_precio - rango * nivel
        # Invertir el 0% y 100%
        niveles['Nivel 0%'] = max_precio
        niveles['Nivel 100%'] = min_precio
    else:
        # Fibonacci normal: comenzando desde el máximo y restando el rango
        for nivel in niveles_fibonacci:
            niveles[f'Nivel {round((nivel)*100,2)}%'] = min_precio + rango * nivel
        # Normal: 100% arriba, 0% abajo
        niveles['Nivel 100%'] = max_precio
        niveles['Nivel 0%'] = min_precio
    
    return niveles




def calcular_percentil(rango, percentil):
    """_summary_
        Calcular el percentil dado un rango 
    Args:
        rango (list): [min, max]
        percentil (int): 1 a 100 donde 50 es el P50

    Returns:
        Valor del percentil
    """
    valor_minimo = rango[0]
    valor_maximo = rango[1]
    return valor_minimo + ((percentil/100) * (valor_maximo - valor_minimo))

def Semaforo_Percentil(df:pd.DataFrame,col_use:str="Volumne",nLag:int=7,rangeCriterio:tuple=(30,70),
                       showPrint=False):
    """_summary_
    Utiliza los ultimos periodos ("nLag") para hacer un intervalo min, max, de tal manera que este intervalo
    se genera una nueva escala de 0 a 100, y colocamos el ultimo valor en esta escala para discrimar en:
    "Alcista", "Bajista", "Neutro".  

    Args:
        df (pd.DataFrame): Contiene la columna a usar y almenos nLag valores
        col_use (str, optional): _description_. Defaults to "Volumne".
        nLag (int, optional): _description_. Defaults to 7.
        rangeCriterio (tuple, optional): _description_. Defaults to (30,70).

    Returns:
        str: (Alcista, Bajista, Neutro)
    """
    #print(df.shape)
    listValue = df[col_use].tail(nLag)
    lastValue = df[col_use].iloc[-1]
    rangeValue = [min(listValue), max(listValue)]
    if showPrint:
        print("LastValue: ",lastValue)
        print("Rango usado: ",rangeValue)
        print("Valor del percentil",rangeCriterio[0], ": ",calcular_percentil(rangeValue, rangeCriterio[0]))
        print("Valor del percentil",rangeCriterio[1], ": ",calcular_percentil(rangeValue, rangeCriterio[1]))
    # Rango
    if lastValue <= calcular_percentil(rangeValue, rangeCriterio[0]) :
        criterio = "Bajista"
    elif lastValue >= calcular_percentil(rangeValue, rangeCriterio[1]) :
        criterio = "Alcista"
    else:
        criterio = "Neutro"
    return criterio


def generate_dict(df,colTrend, col1, col2):
    dict_aux= {}
    for i,j in df[df[colTrend]==0][[col1,col2]].values:
        try:
            dict_aux[i].append(j)
        except:
            dict_aux[i] = [j]
    return dict_aux

def calculate_ichimoku_cloud(df, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, senkou_shift=26):
    """
    Calcula el indicador Ichimoku Cloud y lo agrega al DataFrame dado.
    
    Parámetros:
    df (DataFrame): DataFrame con columnas 'High', 'Low', 'Close'.
    tenkan_period (int): Período para el cálculo de la línea Tenkan-sen.
    kijun_period (int): Período para el cálculo de la línea Kijun-sen.
    senkou_span_b_period (int): Período para el cálculo de la línea Senkou Span B.
    senkou_shift (int): Desplazamiento hacia adelante para las líneas Senkou.

    Retorno:
    DataFrame: DataFrame con las nuevas columnas para las líneas del Ichimoku Cloud.
    """

    # Calculando Tenkan-sen (Línea de Conversión)
    df['Tenkan_sen'] = (df['High'].rolling(window=tenkan_period).max() + df['Low'].rolling(window=tenkan_period).min()) / 2

    # Calculando Kijun-sen (Línea Base)
    df['Kijun_sen'] = (df['High'].rolling(window=kijun_period).max() + df['Low'].rolling(window=kijun_period).min()) / 2

    # Calculando Senkou Span A (Span A)
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(senkou_shift)

    # Calculando Senkou Span B (Span B)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=senkou_span_b_period).max() + df['Low'].rolling(window=senkou_span_b_period).min()) / 2).shift(senkou_shift)

    # Calculando Chikou Span (Span de Retardo)
    df['Chikou_Span'] = df['Close'].shift(-senkou_shift)
    
    # Señales de ichimoku cloud
    # Tenkan-sen (Línea de Conversión ~ corto plazo)
    # Kijun-sen (Línea Base ~ largo plazo)

    ## 1 Color de la nube (Sentimiento de mercado)
    dictColorCloud = {1:"Alcista", -1:"Bajista"}
    df["colorCloud"] = df.apply(lambda row: 1 if row['Senkou_Span_A'] >= row['Senkou_Span_B'] else -1, axis=1)
    ### Contamos el numero de periodos que dura cada uno.
    df["colorCount"] = CountTrend(df["colorCloud"])
    df["colorCloud"] = df["colorCloud"].map(dictColorCloud)
    df['colorCountMax'] = df['colorCount'].shift(1)
    

    ## 2 Distancia o ancho de la nube
    df["distanceCloud"] = abs(df["Senkou_Span_A"] - df["Senkou_Span_B"])

    ## 3 Tendencia marcada por la tenkansen y la kinjun sen.
    df["TrendTenkanKijun"] = df.apply(lambda row: 1 if row['Tenkan_sen'] >= row['Kijun_sen'] else -1, axis=1)
    ### Contamos los periodos que lleva esa tendencia
    df["TrendTenkanKijunCount"] = CountTrend(df["TrendTenkanKijun"])
    df["TrendTenkanKijun"] = df["TrendTenkanKijun"].map({1:"Alcista",-1:"Bajista"}) 
    df['TrendTenkanKijunCountMax'] = df['TrendTenkanKijunCount'].shift(1)
    

    ## 4 Distancia de la linea base vs linea conversion
    df["distanceTenkanKijun"] = abs(df["Tenkan_sen"] - df["Kijun_sen"])

    return df

def detect_sign(x):
    """Esta funcion detecta el signo del valor
    Return:
    1 si es positvo, 0 si es negativo"""
    if isinstance(x, (int, float)):
        if x<=0:
            return 0
        else: return 1
    else:
        raise ValueError(f"El valor de {x}(x) debe ser numerico en la funcion detect_sign")

def CountTrend(serie):
    """
    (Function)
        Esta funcion cuenta el numero de periodos que duro un Trend al timeframe pasado.
        Por lo que returna una serie secuencial de 0 a n, donde n determina la cantidad de periodos que duro el Trend.
    (Parameters)
        serie[iterator] : Datos de los cuales se parten.
    """
    contador = 0
    lcont = []
    for i, val in enumerate(serie):
        
        #print(i)
        if i==0:
            old_val = val
            lcont.append(0)
            continue
        
        
        if detect_sign(val) == detect_sign(old_val):
            contador += 1
        else :
            contador = 0
            
        lcont.append(contador)
        old_val = val
    return lcont

def putTrendDirection(df):
    """_summary_
        Esta funcion Asigna un numero negativo al Trend en caso de short, de lo contrario lo deja igual
    Args:
        df (_type_): Dataframe con columnas Trend (numero entero que cuenta lo que lleva de tendencia) y el "tipo" 
        el cual indica si es short, long or neutral

    Returns:
        list : El cual contiene el nuevo Trend
    """    
    typeOld = "nada"
    newTrend = []
    for index in df.index:
        n = df.loc[index]["Trend"]
        tipo = df.loc[index]["tipo"]
        if str(tipo).lower() == "short":
            n = n*-1
        elif tipo.lower() == "neutral" and typeOld.lower()=="short":
            n = n*-1
                
        typeOld = tipo
        newTrend.append(n)        
    return newTrend        

def CandelIndicators(df):

    df["bodyCandel"] = abs(df["Open"] - df["Close"])
    df["wickupp"] = df.apply(lambda row: row["High"] - max(row["Open"], row["Close"]) , axis=1)
    df["wicklow"] = df.apply(lambda row: min(row["Open"], row["Close"]) - row["Low"]  , axis=1) 
    df["fullCandel"] = df["High"] - df["Low"]
    df["plusCandel"] = df["bodyCandel"] + df["wicklow"] + df["wickupp"]
    df["isBodyTEWick"] = df.apply(lambda row: 1 if row["bodyCandel"]>= max(row["wickupp"], row["wicklow"]) else 0, axis=1)
    probaCandel = dict(round(df["isBodyTEWick"].value_counts().rename({0:"Si", 1:"No"}) /df["isBodyTEWick"].value_counts().sum(axis=0),3))  #.plot(kind="bar",title="% Body Candel Than Equal Wick Candel"))

    # Indicadores de velas
    wickUpMean = df["wickupp"].mean()
    wickLowMean = df["wicklow"].mean()
    bodyMean = df["bodyCandel"].mean()
    fullMean = df["fullCandel"].mean()

    wickUpMedian = df["wickupp"].median()
    wickLowMedian = df["wicklow"].median()
    bodyMedian = df["bodyCandel"].median()
    fullMedian = df["fullCandel"].median()

    # La desviacion se puede usar como el cambio diario, es decir cuanto puede incrementar o decrementar la vela
    wickUpStd = df["wickupp"].std()
    wickLowStd = df["wicklow"].std()
    bodyStd = df["bodyCandel"].std()
    fullStd = df["fullCandel"].std()

    wickUpCV = round(wickUpStd / wickUpMean ,2)
    wickLowCV = round(wickLowStd / wickLowMean ,2)
    bodyCV = round(bodyStd / bodyMean ,2)
    fullCV = round(fullStd / fullMean ,2)

    isBodyTEWick = probaCandel

    typeCandel =  dict(round(df["tipo"].value_counts()/ df["tipo"].value_counts().sum(),4))

    rowCandel = {
        "wickUpMean" : round(wickUpMean,3),
        "wickLowMean" : round(wickLowMean,3),
        "bodyMean" : round(bodyMean,3),
        "fullMean" : round(fullMean,3),
        "wickUpMedian" : round(wickUpMedian,3),
        "wickLowMedian" : round(wickLowMedian,3),
        "bodyMedian" : round(bodyMedian,3),
        "fullMedian" : round(fullMedian,3),
        "wickUpStd" : round(wickUpStd,3),
        "wickLowStd" : round(wickLowStd,3),
        "bodyStd" : round(bodyStd,3),
        "fullStd" : round(fullStd,3),
        "wickUpCV" : round(wickUpCV,3),
        "wickLowCV" : round(wickLowCV,3),
        "bodyCV" : round(bodyCV,3),
        "fullCV" : round(fullCV,3),
        "isBodyTEWick": isBodyTEWick,
        "%typeCandel" : typeCandel
    }
    return rowCandel

def IchimokuIndicator(df, tenkan_period=7, kijun_period=30, senkou_span_b_period=89, senkou_shift=15):
    """
    (Function)
        Esta funcion genera un diccionario que son las señales del ichimoku cloud

    Args:
        df (_type_): _description_
        tenkan_period (int, optional): Corto plazo. Defaults to 7.
        kijun_period (int, optional): MEdiano largo plazo. Defaults to 30.
        senkou_span_b_period (int, optional): Base largo plazo para nube. Defaults to 89.
        senkou_shift (int, optional): Lag. Defaults to 15.

    Returns:
        _type_: _description_
    """
    from Indicadores.Trend import calculate_ichimoku_cloud, generate_dict
    from scipy import stats
    import numpy as np 
    
    df = calculate_ichimoku_cloud(df, tenkan_period, kijun_period, senkou_span_b_period, senkou_shift)

    
    dictColorCloud = generate_dict(df, colTrend="colorCount", col1="colorCloud", col2="colorCountMax")
    dictTrendTenkanKijun = generate_dict(df, colTrend="TrendTenkanKijunCount", col1="TrendTenkanKijun", col2="TrendTenkanKijunCountMax")
    
    alcistaCloud = np.array(dictColorCloud["Alcista"])[~np.isnan(dictColorCloud["Alcista"] )]
    bajistaCloud = np.array(dictColorCloud["Bajista"])[~np.isnan(dictColorCloud["Bajista"] )]

    alcistaTenkanKijun = np.array(dictTrendTenkanKijun["Alcista"])[~np.isnan(dictTrendTenkanKijun["Alcista"] )]
    bajistaTenkanKijun = np.array(dictTrendTenkanKijun["Bajista"])[~np.isnan(dictTrendTenkanKijun["Bajista"] )]

    countCloudMeanBajista = np.mean(bajistaCloud)
    countCloudMeanAlcista = np.mean(alcistaCloud)
    countCloudModeBajista = stats.mode(bajistaCloud)
    countCloudModeAlcista = stats.mode(alcistaCloud)

    countTenkanKijunMeanBajista = np.mean(bajistaTenkanKijun)
    countTenkanKijunMeanAlcista = np.mean(alcistaTenkanKijun)
    countTenkanKijunModeBajista = stats.mode(bajistaTenkanKijun)
    countTenkanKijunModeAlcista = stats.mode(alcistaTenkanKijun)

    return {"DurationCloudTrend": df["colorCount"].iloc[-1],
    "TypeCloudTrend": df["colorCloud"].iloc[-1],
    "DistanceCloud": df["distanceCloud"].iloc[-1],
    "DistanceCloud%":df["distanceCloud"].iloc[-1] / df["Close"].iloc[-1],
    "MaxCloudAlcista":alcistaCloud,
    "MeanCloudAlcista": countCloudMeanAlcista,
    "ModeCloudAlcista": countCloudModeAlcista,
    "MaxCloudBajista":bajistaCloud,
    "MeanCloudBajista": countCloudMeanBajista,
    "ModeCloudBajista": countCloudModeBajista,
    "DurationTenkanKijunTrend": df["TrendTenkanKijunCount"].iloc[-1],
    "TypeTenkanKijunTrend": df["TrendTenkanKijun"].iloc[-1],
    "DistanceTenkanKijun": df["distanceTenkanKijun"].iloc[-1],
    "TrendTenkanKijunCount%":df["distanceTenkanKijun"].iloc[-1] / df["Close"].iloc[-1], 
    "MaxTenkanKijunAlcista": alcistaTenkanKijun,
    "MeanTenkanKijunAlcista": countTenkanKijunMeanAlcista,
    "ModeTenkanKijunAlcista": countTenkanKijunModeAlcista,
    "MaxTenkanKijunBajista": bajistaTenkanKijun,
    "MeanTenkanKijunBajista": countTenkanKijunMeanBajista,
    "ModeTenkanKijunBajista": countTenkanKijunModeBajista}
    
def StatisticIndicator(df, col_use, periodo_std=0):
    '''
    (Function)
        Esta funcion devuelve informacion estadistica de un dataFrame
    (Parameters)
        - df : El dataframe a utilizar
        - col_use: la columna a estudiar
        - periodo_std: ultimos n-registros lo cuales se toman de muestra para el estudio, si es 0, se toma el total del dataframe
    (Returns)
        Dict cuya key es el indicador estadistico y el valor
    '''
    if periodo_std != 0:
        df = df.tail(periodo_std)
        
    df["rend_per"] = df[col_use].pct_change(1).shift(-1)*100
    df["dy"] = df[col_use].diff(periods=1)
    df["Trend"] = CountTrend(df["dy"])
    
    price_now = df[col_use].iloc[-1]
    price_max = df[col_use].max()
    price_min = df[col_use].min()
    price_dif_max = price_max - price_now
    price_dif_min = price_now - price_min
    probaTrend = round(df["Trend"].value_counts()/df["Trend"].value_counts().sum(),4).to_dict()
    price_percent_max = round(price_dif_max/price_now*100,3)
    price_percent_min = round(price_dif_min/price_now*100,3)
    std_rend = round(df["rend_per"].std(),5)
    mean_rend = round(df["rend_per"].mean(),3)
    
    
    return { "%DownPrice": price_percent_min,
            "%UpPrice": price_percent_max,
            "MeanRend": mean_rend,
            "MeanVolume":df["Volumne"].mean(),
            "StdRend": std_rend,
            "CoefVarRend":round(std_rend/mean_rend,3),
            "ProbaTrendPeriod" : probaTrend,
            }

def GenerateInfoAll(cliente,interval,date_start='2022-01-01 00:00:00',col_use="Close",onlyUSDT=True,
                        tope=10000000,printExcept=True,medias_default=[7,30], svo_default=20, 
                        periodos_macd=[7,30],signal_macd=9,periodos_bb=20, std_bb=2, tipo_bb="mean_reversion",
                        period_RSI=12,rsi_zones=[20,80], periodos_stochastic=[14,3],periodo_std=0):
    
    from Datasets import  gethistorical_trades, generate_dataset
    from tqdm import tqdm
    import pandas as pd 
    from scipy import stats
    import traceback
    medias_orig = medias_default
    mm = Medias_Moviles()
    svo = svo_default
    
    format_date_start = '%Y-%m-%d %H:%M:%S' # Poner conforme al date_start, en caso de ser auto, no mover
    type_upload = 'manual'
    list_full = []

    #df_simbolos = pd.read_csv("Data/Simbolos_UMFutures.csv")
    ruta_EMAs = r"Data/Historicos/Resultados_MyClass.csv"
    df_simbolos = pd.read_csv(ruta_EMAs)
    all_simbols = df_simbolos[(df_simbolos["symbol"].str.contains("USDT")) & (df_simbolos["EMA_corta"] != "Sin Datos suficientes")].drop_duplicates(subset=["symbol","interval"])
    contador = 0
    novalidos = []
    df_emas = pd.read_csv("./Data/Resultado.csv",header=None)

    df_btc = generate_dataset(cliente, "BTCUSDT", interval,
                            date_start,fecha_format=format_date_start,
                            type_upload=type_upload,
                            con=None,table_name=None) 

    for sim in tqdm(all_simbols["symbol"].drop_duplicates()):
        if onlyUSDT:
            if not "USDT" in sim:
                continue
        try:
            if printExcept:
                print("Entro al Try")
            df = generate_dataset(cliente, sim, interval,
                            date_start,fecha_format=format_date_start,
                            type_upload=type_upload,
                            con=None,table_name=None)
            if sim == "BTCUSDT":
                df_btc = df.copy()
            df["tipo"] = df.apply(lambda row: "long" if row["Open"]< row["Close"] else "neutral" if row["Close"] == row["Open"] else "short", axis=1)
            df["rend_per"] = df[col_use].pct_change(1).shift(-1)*100
            df["dy"] = df[col_use].diff(periods=1)
            df["Trend"] = CountTrend(df["dy"])
            if len(df_simbolos[(df_simbolos["symbol"]==sim) & (df_simbolos["interval"].isin([interval]))])==1:
                 medias = [int(df_simbolos.loc[(df_simbolos["symbol"]==sim) & (df_simbolos["interval"].isin([interval])),"EMA_corta"].values[0]), int(df_simbolos.loc[(df_simbolos["symbol"]==sim) & (df_simbolos["interval"].isin([interval])),"EMA_larga"].values[0])]
                 svo = int(df_simbolos.loc[(df_simbolos["symbol"]==sim) & (df_simbolos["interval"].isin([interval])),"SVO"].values[0])
            elif len(df_simbolos[(df_simbolos["symbol"]==sim) & (df_simbolos["interval"].isin([interval]))])>1:
                raise NameError(f"Hay mas de un valor con el mismo simbolo e intervalo en {ruta_EMAs}")
            else:
                medias = medias_orig
                
            print(medias)
            #periodos_ema = [7,29]
            df = mm.Colocar_Medias_Moviles(df1=df,columnas={"precio":col_use,"volumen":"Volumne"}, periodos_mm=medias)
            
            # calculamos la distancia entre la ema7 y la ema 28
            df["distancia_EMA"] = df[f"EMA_{medias[0]}"] - df[f"EMA_{medias[1]}"]
            #df["distancia_SMA"] = df["SMA0"] - df["SMA1"]
            # Contamos el numero de dias en TRENd ema
            df["EMATrend"] = CountTrend(df["distancia_EMA"])
            #df["SMATrend"] = CountTrend(df["distancia_SMA"])
            
            # Sobre el MACD
            df = mm.calcular_MACD(df, col_use="Close",fastLength=medias[0], slowLength=medias[1], signalLength=9)
            df["distancia_MACD"] = df["Signal_Line"] - df["MACD"]
            df["MACDTrend"] = CountTrend(df["distancia_MACD"])
            
            # EstrategiaEMAS
            e = Estrategias()
            resultados = []
            fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharey=False, sharex="col")  #5, 1.7 mean_reversion  breakout
            cap_final_mm = e.Plot_Estrategia_MM_SVO(df, axes, sim, col_MM="EMA", periodos_mm=medias, period_SVO=svo, put_SVO=False, capital_inicial=100, comision=0.001, show_print=False,
                                        salida_SVO=False)
            cap_fin_mm = cap_final_mm[0]
            
            
            cap_final_macd = e.Plot_Estrategia_MACD(df, axes, sim, period_señal=signal_macd, periodos_macd=periodos_macd, period_SVO=svo,
                            capital_inicial=100, comision=0.001, show_print=False, put_SVO=True,  
                            put_plot=False)
            
            cap_final_bb = e.Plot_Estrategia_BB(df, axes, sim, periodos_bb=periodos_bb, std_bb=std_bb,
                                                tipo=tipo_bb, capital_inicial=100, comision=0.001, 
                                                show_print=False, put_plot=False)
            
            cap_final_rsi = e.Plot_Estrategia_RSI(df, axes, sim, period_RSI=period_RSI, period_SVO=svo, periodos_mm=medias,
                            capital_inicial=100, comision=0.001, show_print=False, put_SVO=True,  
                            rsi_zones=rsi_zones, put_plot=False, col_MM="EMA") 
            
            cap_final_estocastico = e.Plot_Estrategia_Stocastico(df, sim, periodos_stochastic=periodos_stochastic, period_SVO=svo, periodos_mm=medias,
                            capital_inicial=100, comision=0.001, show_print=False, put_SVO=True,  axes=None,
                            rsi_zones=rsi_zones, put_plot=False, col_MM="EMA")
            
            
            if printExcept:
                print("Paso las Columns")
                print(Semaforo_Percentil(df=df,col_use="Volumne", nLag=medias[0], rangeCriterio=(75,25)))
            
            
            price_now = df[col_use].iloc[-1]
            price_max = df[col_use].max()
            price_min = df[col_use].min()
            
            
            
            
            row = {"symbol": sim,
            f"lastPrice{col_use}": price_now,
            "lastVolume":df["Volumne"].iloc[-1],
            "MaxPrice": price_max,
            "MinPrice": price_min,
            
            "TimeLastCrossEMA's": df["EMATrend"].iloc[-1],
            "TimeLastCrossMACD":cap_final_macd[-1],
            "TimeLastCrossSMA's": df["SMATrend"].iloc[-1],
            "TimeLastCrossBB_INF": cap_final_bb[-1],
            "TimeLastCrossBB_SUP": cap_final_bb[-2],
            
            "CapitalFinal_MM":cap_fin_mm,
            "CapitalFinal_MACD":cap_final_macd[0],
            "CapitalFinal_BB": cap_final_bb[0],
            "CapitalFinal_RSI":cap_final_rsi[0],
            "CapitalFinal_Stochastic":cap_final_estocastico[0],
            
            "SVO":svo,
            "EMA_Corta" : medias[0],
            "EMA_Larga" : medias[1],
            "TrendEMA": "Alcista" if df["distancia_EMA"].iloc[-1]>= 0 else "Bajista",
            "Señal_MACD" : signal_macd,
            "Periodos_macd" : periodos_macd,
            "Periodos_bb":periodos_bb,
            "Std_bb":std_bb,
            "Tipo_bb":tipo_bb,
            "Periodo_RSI": period_RSI,
            "Zonas_RSI": rsi_zones,
            "Periodos_Estocastico":periodos_stochastic,
            
            
            "CorrelationBTC": round(df_btc[col_use].corr(df[col_use]),4),
            "N":df.shape[0]
            }
            
            """
            
            "TrendSMA": "Alcista" if df["distancia_SMA"].iloc[-1]>= 0 else "Bajista",
            "TrendMACD": "Alcista" if df["distancia_MACD"].iloc[-1] <=0 else "Bajista" ,
            f"TrendVol{medias[0]}":Semaforo_Percentil(df=df,col_use="Volumne", nLag=medias[0], rangeCriterio=(70,30)),
            f"TrendVol{medias[1]}":Semaforo_Percentil(df=df,col_use="Volumne", nLag=medias[1], rangeCriterio=(70,30)),            
            "Trend":df["tipo"].iloc[-1],
            """  
            
            if printExcept:
                print("Genero row")
            rowCandel = CandelIndicators(df)
            rowStd = StatisticIndicator(df, col_use, periodo_std)
            if printExcept: print("Genero rowCandel")
            #rowIchimoku = IchimokuIndicator(df)
            #if printExcept: print("Genero rowIchimoku")
            list_full.append({**row, **rowStd}) #**rowCandel,**rowIchimoku})
            
            print("paso todo")
            contador += 1
            if contador >= tope:
                break
            
        except Exception as e:
            if printExcept:
                print("----------Error-------------")
                print(e)
                traceback.print_exc()
            novalidos.append(sim)
            if len(novalidos) >= tope:
                break
            
            
        
    #print(list_full)
    print("---****---***---*-*-*-*-*-*-*")
    print("No validos fueron: ", len(novalidos))
    print("Validos: ",contador)
    #print({**row, **rowIchimoku, **rowCandel})
    df_final = pd.DataFrame(list_full)
    #df_final["N"] = df_final["Shape"].map(lambda x: x[0])
    print(df_final.columns)
    df_final.to_csv(f"./Data/InfoSimbolos_Spot{interval}.csv", index=False)
    print("Se actualizo el archivo: ",f"InfoSimbolos_Spot{interval}.csv")
    return None 