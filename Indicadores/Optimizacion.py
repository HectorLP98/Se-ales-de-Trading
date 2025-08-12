import itertools
from multiprocessing import Pool, cpu_count
#from Indicadores.Estrategias import Estrategias
from tqdm import tqdm
from Indicadores.Estrategias import *
import matplotlib.pyplot as plt
#from Estrategias import Estrategias



# Función para probar una combinación de parámetros
def probar_estrategia_MACD(params):
    e = Estrategias()  # Crear una nueva instancia de la clase Estrategias
    #fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=False)
    df, sim, macd, svo, señal = params
    
    # Ejecutar la estrategia con los parámetros actuales
    cap_final = e.Plot_Estrategia_MACD(df, axes=None, symbol=sim, period_señal=señal, periodos_macd=macd, period_SVO=svo,
                            capital_inicial=100, comision=0.001, show_print=False, put_SVO=True,  
                            put_plot=False)
    #plt.show()
    # Devolver el capital final y los parámetros utilizados
    return cap_final


# Función para probar una combinación de parámetros
def probar_estrategia_MM_SVO(params):
    e = Estrategias()  # Crear una nueva instancia de la clase Estrategias
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=False)
    df, sim, medias, svo = params
    
    # Ejecutar la estrategia con los parámetros actuales
    cap_final = e.Plot_Estrategia_MM_SVO(df, axes, sim, col_MM="EMA", periodos_mm=medias, period_SVO=svo, put_SVO=False, 
                                         capital_inicial=100, comision=0.001, show_print=False, salida_SVO=False)
    #plt.show()
    # Devolver el capital final y los parámetros utilizados
    return cap_final[0], medias, svo

# Función para probar una combinación de parámetros
def probar_estrategia_RSI(params):
    e = Estrategias()  # Crear una nueva instancia de la clase Estrategias
    #fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=False)
    try:
        df, sim, rsi, svo, medias = params
        put_svo = False
    except:
        df, sim, rsi, medias = params
        put_svo = True
    
    # Ejecutar la estrategia con los parámetros actuales
    cap_final = e.Plot_Estrategia_RSI(df, None, sim, period_RSI=rsi, period_SVO=svo, periodos_mm=medias,
                                      capital_inicial=100, comision=0.001, show_print=False, put_SVO=put_svo, 
                                      rsi_zones=[20,80], put_plot=False)
    
    #plt.show()
    # Devolver el capital final y los parámetros utilizados
    return cap_final[0], rsi, svo, medias

def probar_estrategia_MACD_Estocastico(params):
    e = Estrategias()
    try:
        df, indicadores,ptj_ganar,relacion = params
    except:
        ValueError("PArametros incompletos")
    
    cap_final = e.cerebro(df,estrategia="macd_estocastico",indicadores=indicadores,
          putPlot=False,axes=None,ptj_ganar=ptj_ganar,relacion=relacion)
    return cap_final

def optimizar_estrategia_rsi(df, sim, rango_rsi, rango_svo,rango_medias):
    # Crear una lista de combinaciones de parámetros para probar
    combinaciones = list(itertools.product(rango_rsi, rango_svo, rango_medias))
    
    # Preparar los parámetros para pasar al pool de procesos
    parametros = [(df, sim, rsi, svo, medias) for rsi, svo, medias in combinaciones]
    
    # Usar multiprocesamiento con tqdm para mostrar barra de progreso
    with Pool(cpu_count()) as pool:
        resultados = list(tqdm(pool.imap(probar_estrategia_RSI, parametros), total=len(parametros)))
    
    # Encontrar el resultado con el máximo capital
    mejor_resultado = max(resultados, key=lambda x: x[0])  # El capital está en la primera posición del tuple
    
    # Imprimir el mejor resultado
    #print(f"Mejor capital: {mejor_resultado[0]} con : {mejor_resultado[1]} y SVO: {mejor_resultado[2]}")
    
    return mejor_resultado

# Función para optimizar usando multiprocesamiento
def optimizar_estrategia0(df, sim, rango_medias, rango_svo):
    # Crear una lista de combinaciones de parámetros para probar
    combinaciones = list(itertools.product(rango_medias, rango_svo))
    
    # Preparar los parámetros para pasar al pool de procesos
    parametros = [(df, sim, medias, svo) for medias, svo in combinaciones]
    
    # Usar multiprocesamiento para probar todas las combinaciones
    with Pool(cpu_count()) as pool:
        resultados = pool.map(probar_estrategia, parametros)
    
    # Encontrar el resultado con el máximo capital
    mejor_resultado = max(resultados, key=lambda x: x[0])  # El capital está en la primera posición del tuple
    
    # Imprimir el mejor resultado
    print(f"Mejor capital: {mejor_resultado[0]} con medias: {mejor_resultado[1]} y SVO: {mejor_resultado[2]}")
    
    return mejor_resultado


# Función para optimizar usando multiprocesamiento con barra de progreso
def optimizar_estrategia_MM_SVO(df, sim, rango_medias, rango_svo):
    # Crear una lista de combinaciones de parámetros para probar
    combinaciones = list(itertools.product(rango_medias, rango_svo))
    # Preparar los parámetros para pasar al pool de procesos
    parametros = [(df, sim, medias, svo) for medias, svo in combinaciones]
    # Usar multiprocesamiento con tqdm para mostrar barra de progreso
    with Pool(cpu_count()) as pool:
        resultados = list(tqdm(pool.imap(probar_estrategia_MM_SVO, parametros), total=len(parametros)))
    # Encontrar el resultado con el máximo capital
    mejor_resultado = max(resultados, key=lambda x: x[0])  # El capital está en la primera posición del tuple
    return mejor_resultado

def optimizar_estrategia_MACD(df, sim, rango_macd, rango_svo, rango_señal):
    # Crear una lista de combinaciones de parámetros para probar
    combinaciones = list(itertools.product(rango_macd, rango_svo, rango_señal))
    # Preparar los parámetros para pasar al pool de procesos
    parametros = [(df, sim, macd, svo, señal) for macd, svo, señal in combinaciones]
    # Usar multiprocesamiento con tqdm para mostrar barra de progreso
    with Pool(cpu_count()) as pool:
        resultados = list(tqdm(pool.imap(probar_estrategia_MACD, parametros), total=len(parametros)))
    # Encontrar el resultado con el máximo capital
    mejor_resultado = max(resultados, key=lambda x: x[0])  # El capital está en la primera posición del tuple
    return mejor_resultado
