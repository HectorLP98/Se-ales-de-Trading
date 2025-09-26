# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:29:13 2022

@author: 52551
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from Datos.fechas import fecha_a_milisegundos, interval_to_milliseconds
import tqdm



def get_historical_from(cliente, symbol:str, interval:str, 
                        date_start:str,fecha_format: str = '%Y-%m-%d %H:%M:%S',
                        is_yf=False):
    '''
    (Function)
    Esta funcion integra la recursividad para extraer el historico de un symbolo 
    en binance
    (Parameters)
        - cliente: [binance.client] cliente para poder consumir la api
        - symbol: [str] Indica el activo (par relacional de criptoactivos) a extraer, ejemplo  (BTCUSDT)
        - interval: [str]  Indica el intervalo del historico
        - date_start [str] Fecha de inicio de la extracción, ejemplo "2022-01-01 00:00:00" poner formato en caso
          de modificar el formato de fecha
        - fecha_format: [str] ejemplo del formato '%Y-%m-%d %H:%M:%S'
    (Returns)
        lista de precios, cuyo elemento es un diccionario con los datos en corte horizontal
        
    '''
    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)
    limite = 1000
    idx = 0
    # init our list
    output_data = []
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    start_ts = fecha_a_milisegundos(date_start,fecha_format) #datetime.strptime(date_start, format_date_start)
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if set
        if not is_yf:
            temp_data = cliente.klines(symbol=symbol, interval=interval, # Aternativa get_klines
                                limit=limite, startTime=start_ts ) 
        else:
            temp_data = cliente.history(interval=interval, start=date_start)
        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_data):
            symbol_existed = True

        if symbol_existed:
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limite:
            # exit the while loop
            break
        
        if idx % 50 == 0:
            print("Llevamos ",idx, " iteraciones")
        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)
    return output_data


def generate_dataset(cliente, symbol, interval,
                     date_start,fecha_format='%Y-%m-%d %H:%M:%S',
                    type_upload='manual',
                    hrs_diff=6,con=None,table_name=None, version_ldate="local"):
    '''
    (Function)
        Esta funcion genera un DataFrame con los datos historico de un simbolo que 
        opera en binance.
    (Parameters)
        - cliente: [binance.client] cliente para poder consumir la api
        - symbol: [str] Indica el activo (par relacional de criptoactivos) a extraer, ejemplo  (BTCUSDT)
        - interval: [str]  Indica el intervalo del historico
        - date_start [str] Fecha de inicio de la extracción, ejemplo "2022-01-01 00:00:00" poner formato en caso
          de modificar el formato de fecha
        - fecha_format: [str] ejemplo del formato '%Y-%m-%d %H:%M:%S'
        - type_upload: [str] debe ser ["auto", "manual"] por default es "manual",
            determina la manera de extraer los datos. Si desea que identifique la fecha por 
            si solo use "append" y necesitara incluir el conector (con) y el nombre de la 
            tabla (table_name) para extraer los datos de dicha base.
            En caso de usar manual necesitara solo la fecha apartir de la cual quiere los datos.
        - hrs_diff: [int] Debido al UTC es necesario quitar horas para emparejar a nuestra hora
                segun el pais, puede hacer pruebas para ajustar a su pais, default 6
        - con: Conector con la base de datos que contiene sus historicos, solo si type_upload="auto" 
        - table_name: [str] NOmbre de la tabla del historico a analisar, solo si type_upload="auto" 
        - version_ldate: Establecela como "local" en caso de usar visual_studio, si no jala por fecha
          usar foreign.
    (Return)
        pd.DataFrame
    '''
    if type_upload.lower() == 'manual':
        
        if isinstance(date_start,str):
            ## Seteamos a datetime y ajustamos el tiempo por zona horaria
            last_date = datetime.strptime(date_start, fecha_format)
            
        else:
            raise ValueError('date_start debe ser un str, example 2020-01-01 00:00:00')
        
        
    elif type_upload.lower() == "auto":
        
        # Leemos la ultima fecha para cargar los datos apartir de ahi.
        # Close_Time, Hour_Close
        last_df = pd.read_sql_query(f'''
                    SELECT Close_Time
                    FROM {table_name}
                    order by Close_Time desc
                    LIMIT 1
                    ''', con)
        if version_ldate.lower() == "local":
            last_date = last_df.Close_Time.dt.strftime(fecha_format).values[0]
        else:
            date_str = last_df.Close_Time.values[0]
            date_str = date_str.split(".")[0]
            last_date = datetime.strptime(date_str, fecha_format)
        
        
    else:
        raise ValueError("type_upload musth be in [auto,manual]")
    
    # Hacemos el request a la api de binance
    output_data = get_historical_from(cliente,symbol,interval,last_date,fecha_format)
    #Damos formato DataFrame
    doc_columns = ['Open_Time','Open','High','Low','Close','Volumne',
                'Close_Time','Quote_asset_vol','Number_trades','Taker_buy_base',
                'Taker_buy_quote','Ignore']
    df_aux = pd.DataFrame(output_data,columns=doc_columns)
    
    for col in ["Open","High",	"Low","Close","Volumne","Taker_buy_base","Taker_buy_quote",
            "Quote_asset_vol"]:
        df_aux[col] = df_aux[col].astype(float)
                
    for col in ["Number_trades"]:
        df_aux[col] = df_aux[col].astype(int)
        
    df_aux['Open_Time'] = pd.to_datetime(df_aux['Open_Time'],unit = 'ms' ) -  timedelta(hours=hrs_diff)
    df_aux['Close_Time'] = pd.to_datetime(df_aux['Close_Time'],unit = 'ms' ) -  timedelta(hours=hrs_diff)
    
    return df_aux


def up_to_db(cliente, simbolos:list, intervalos:list, 
             date_start:str,fecha_format:str='%Y-%m-%d %H:%M:%S',
             type_upload='manual',
             hrs_diff=6,con_get=None,con_set=None,if_exist='replace',version_ldate='local'):
    '''
    (Function)
        Esta funcion genera un DataFrame con los datos historico de un simbolo que 
        opera en binance.
    (Parameters)
        - cliente: [binance.client] cliente para poder consumir la api
        - simbolos: [list|tuple|iterable] Indica los activos (par relacional de criptoactivos) a extraer, ejemplo  (BTCUSDT)
        - intervalos: [list|tuple|iterable]  Indica los intervalos del historico
        - date_start [str] Fecha de inicio de la extracción, ejemplo "2022-01-01 00:00:00" poner formato en caso
          de modificar el formato de fecha
        - fecha_format: [str] ejemplo del formato '%Y-%m-%d %H:%M:%S'
        - type_upload: [str] debe ser ["auto", "manual"] por default es "manual", determina la manera de extraer los datos. Si desea que identifique la fecha por 
            si solo use "append" y necesitara incluir el conector (con) y el nombre de la 
            tabla (table_name) para extraer los datos de dicha base.
            En caso de usar manual necesitara solo la fecha apartir de la cual quiere los datos.
        - hrs_diff: [int] Debido al UTC es necesario quitar horas para emparejar a nuestra hora
                segun el pais, puede hacer pruebas para ajustar a su pais, default 6
        - con: Conector con la base de datos que contiene sus historicos, solo si type_upload="auto" 
        - if_exist: [str] por default "replace" puede ser ['fail', 'replace', 'append'], indica 
                    que hacer en caso de que la tabla que intenta cargar exista.
        - version_ldate: Establecela como "local" en caso de usar visual_studio, si no jala por fecha
          usar foreign.
    (Return)
        pd.DataFrame
    '''
    if isinstance(simbolos,str): 
        simbolos = [simbolos]
    if isinstance(intervalos,str):
        intervalos = [intervalos]
        
    for symbol in tqdm.tqdm(simbolos):
        for interval in intervalos:
            
            # NOmbre de la tabla
            table_name = symbol+'_'+interval
            
            df = generate_dataset(cliente, symbol, interval,
                        date_start,fecha_format=fecha_format,
                        type_upload=type_upload,
                        hrs_diff=hrs_diff,con=con_get,table_name=table_name,
                        version_ldate=version_ldate)
                

            df = df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volumne', 'Close_Time',
                'Quote_asset_vol', 'Number_trades', 'Taker_buy_base', 'Taker_buy_quote']]

            
            revisar = df.iloc[-1].values
            cumplen = np.where(revisar==0)
            if len(cumplen[0])>1:
                df = df.iloc[:-1]
            cargados = df.to_sql(table_name,con_set,if_exists=if_exist,index=False)

            print(f"Se cargo la tabla: {table_name}")
            time.sleep(1)


# PRofundidad de mercado
def depth(cliente,simbolo,limite):
# from perfil_binance import cuenta_binance as cb
# client = Client('', '')
# cliente = cb('demo')
# simbolo = 'TRXUSDT'
# limite = 100 hasta 5,000
# return un df con la cantidad de dolares y el precio del activo.

    
    depth = cliente.get_order_book(symbol=simbolo, limit=limite)
    lp_c = []
    lp_v = []
    l_c = []
    l_v = []
    for key, val in depth.items():
        if key == 'bids': # Compras
            for j in val:
                lp_c.append(float(j[0]))
                l_c.append(round(float(j[1]),2))
           # print(key,len(val))
        elif key == 'asks': # Ventas
           # print(key,len(val))
            for j in val:
               # print(j)
                lp_v.append(float(j[0]))
                l_v.append(round(float(j[1]),2))
    df = pd.DataFrame({'Q_venta': l_v,
                       'P_venta': lp_v,
                       'Q_compra':l_c,
                       'P_compra':lp_c})
    df.tail()
    return df



# isBuyerMaker: true => la operación fue iniciada por el lado de la venta; el lado de la compra ya era el libro de pedidos. es compra
# isBuyerMaker: false => la operación fue iniciada por el lado comprador; el lado de la venta ya era el libro de pedidos     es venta                                                                          False = orden de mercado. no pasa por el libro.
# qty: cantidad de cripto 
# quoteQty: Total de compra en USDT



def gethistorical_trades(cliente,simbolo, agomin=0, agohr=0):


    '''
    (Function)
        Esta funcion devuelve el historial de ordenes ejecutadas en el exchenge.
    (PArameters)
        - cliente [Spot()]: De binance
        - simbolo [str]: De interes para obtener [BTCUSDT,CITYUSDT]
        - agomin [int]: Numero de minutos atras para visualizar
        - agohr [int]: Numero de horas atras para visualizar
    (Return)
        - DataFrame.
'''  
    
    trades = cliente.historical_trades(symbol=simbolo, limit=1000 )# , fromId =dfp.id.min()-1000)
    date_min  = min(d['time'] for d in trades)
    date_min = datetime.fromtimestamp(date_min/1000)
    tb = date_min - timedelta(minutes = agomin) - timedelta(hours=agohr)
    
   # print(dfp.shape, '*********')
    #print('Time buscado ', tb)
    while date_min>tb:
        trades_Aux = cliente.historical_trades(symbol=simbolo, limit=1000 , fromId =min(d['id'] for d in trades)-1000)
        #dfp.time = df.time - timedelta(hours=5)
        date_min =  min(d['time'] for d in trades)
        date_min = datetime.fromtimestamp(date_min/1000)
        trades += trades_Aux
        
    dfp = pd.DataFrame(trades)
    dfp['time'] = pd.to_datetime(dfp['time'],unit = 'ms' )
    dfp.time = dfp.time - timedelta(hours=6)
    print("Total de trades cargados: ",dfp.shape[0])
    date_min = dfp.time.min()
    #dfp['hour'] = dfp["time"].dt.hour
    #dfp['minute'] = dfp["time"].dt.minute
    dfp['T'] = dfp['time'].dt.round('T') 
    #print(dfpp.time.min())
    dfp = dfp.append(dfp, ignore_index = True)
    dfp = dfp.astype({"price":float, "qty":float, "quoteQty":float})
    df = dfp.groupby(by=["T","isBuyerMaker"],as_index=False).agg({"price":"sum", "qty":sum, 'quoteQty':sum})
    df["Total"] = df.apply(lambda row: row["quoteQty"] if row["isBuyerMaker"]==True else -row["quoteQty"] , axis=1)
    
    return df



def Get_Sets_Hist(cliente,symbol,interval,periodos=999,use_binance=True,return_split=False):
    """_summary_

    Args:
        cliente (_type_): Necesario cuando use_binance sea verdadero
        symbol (_type_): Nombre del activo tal cual aparece en Binance(ej:BTCUSDT,BNBUSDT,...) Yahoo(BTC-USD,NQ=F )
        interval (_type_): 1h, 1d, etc
        periodos (int, optional): Cantidad de  periodos a considerar. Defaults to 999., tambien puede usar un strin con el formato: 'AAAA-MM-DD 00:00:00' (ej: '2020-01-01 00:00:00')
        use_binance (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    #import yfinance as yf
    def tiempo_vs_intervalo(periodos:int,interval:str):
        ("1m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1W","1M")
        daysAgo = 0
        hoursAgo = 0
        if interval in ("1m"):
            daysAgo = periodos/(60*24)
        elif interval.lower() in ("5m"):
            daysAgo = periodos/144
        elif interval.lower() in ("15m"):
            daysAgo = periodos/96
        elif interval.lower() in ("30m"):
            daysAgo = periodos/48
        elif interval.lower() in ("1h"):
            daysAgo = periodos/24
        elif interval.lower() in ("2h"):
            daysAgo = periodos/12
        elif interval.lower() in ("4h"):
            daysAgo = periodos/6
        elif interval.lower() in ("6h"):
            daysAgo = periodos/4
        elif interval.lower() in ("8h"):
            daysAgo = periodos/3
        elif interval.lower() in ("12h"):
            daysAgo = periodos/2
        elif interval.lower() in ("1d"):
            daysAgo = periodos*1
        elif interval.lower() in ("3d"):
            daysAgo = periodos/3
        elif interval.lower() in ("1w"):
            daysAgo = periodos*7
        elif interval in ("1M"):
            daysAgo = periodos*30
        
                
        ago = str(datetime.now() - timedelta(days=daysAgo+1))[:10]+" 00:00:00"
        #print("Ago: ",ago)
        return ago
    
    type_upload = 'manual' # ['auto','manual']
    if isinstance(periodos,(int,float)):
        try:
            date_start = tiempo_vs_intervalo(periodos,interval) 
        except:
            raise ValueError("Ingreso un tipo de periodos invalido, aseguerese de usar un entero no multiplo de 100")
    elif isinstance(periodos,str):
        try:
            fecha = periodos.split(" ")[0]
            hora = periodos.split(" ")[1]
            nfecha = True if len(fecha.split("-"))==3 else False
            nhora = True if len(hora.split(":"))==3 else False
            if nfecha==False or nhora==False:
                raise ValueError(f"Formato El parametro ago debe tener el formato YYYY-MM-DD HH:mm:ss, ejemplo; 2024-05-10 00:00:00: {periodos}")
            date_start = periodos
        except:
            raise ValueError(f"Except: El parametro ago debe tener el formato YYYY-MM-DD HH:mm:ss, ejemplo; 2024-05-10 00:00:00 {periodos}")
    else:
        raise ValueError(f"Debe definir el valor de periodos como un entero o formato str de fecha aceptable")
    format_date_start = '%Y-%m-%d %H:%M:%S' # Poner conforme al date_start, en caso de ser auto, no mover
    table_name = symbol +'_'+ interval

#    ans_tipo = input("Usa binace?:  ").lower()

    if use_binance in (1, "1","si","yes","claro",True):
        df = generate_dataset(cliente, symbol, interval,
                            date_start,fecha_format=format_date_start,
                            type_upload=type_upload,
                            con=None,table_name=None)
    else :
        df = yf.download(symbol) #,start= datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S'), end=datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')+timedelta(days=60), interval=interval)

    df.Name = table_name
    if use_binance in (1, "1","si","yes","claro"):
        df["ct"] = df.Close_Time #+ timedelta(seconds=1)
        # Hacer index la fecha
        df = df.set_index("Close_Time",drop=True)
        # Borramos col ignore
        df.drop(columns='Ignore',inplace=True)

    else:
        df["ct"] = pd.to_datetime(df.index)
        df.rename(columns={"Volume":"Volumne"}, inplace=True)
        
    # Crear nuevas columnas
    df['dia_num'] = df['ct'].dt.day               # Día numérico del mes
    df['mes'] = df['ct'].dt.month                 # Mes numérico
    df['dia_mes'] = df['ct'].dt.strftime('%d%m')  # Concatenación día y mes como un string
    df['fecha'] = df['ct'].dt.strftime('%d%m%Y')
    #print(df.shape)
    df = df[df["ct"]>=date_start]
    df["Volume"] = df["Volumne"]
    
    if return_split:
        n_datos = len(df)

        # Calcular el índice de corte para el 80% de los datos
        split_index = int(n_datos * 0.8)

        # Crear los conjuntos de entrenamiento y prueba
        df_train = df.iloc[:split_index].copy()  # 80% para entrenamiento
        print("Fecha rango Train:  ", df_train.index.min(), df_train.index.max())
        df_test = df.iloc[split_index:].copy()    # 20% para prueba
        print("Fecha rango Test:  ", df_test.index.min(), df_test.index.max())
        print(f"Shape train train: {df_train.shape}\nShape test: {df_test.shape}")
        print(f"Total de periodos descargados: {df.shape[0]} de {interval}")
        return df_train,df_test
    else: return df
