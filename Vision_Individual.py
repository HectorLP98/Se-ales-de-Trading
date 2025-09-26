from datetime import datetime, timedelta
import streamlit as st
from binance.spot import Spot as Client

from GraficasPloty import * #graficar_velas, graficar_linea, 
from Std_streamlit import  preparar_dataset,  Poner_Indicadores, AnalizadorIndicadores
from Indicadores.DominioPropio import detectar_swing_points, Contar_FVGs
from Inputs.Simbolos import Obtener_Simbolos_Binance
from Inputs.Requerimentos_streamlit import Solicita_Parametros_Indicadores
from Indicadores.AccioPrecio import AnalizadorVelas
from Estrategias.DeepDeterministicPolicyFradient import *

# Configuración general
st.set_page_config(layout="wide", page_title="Vision Individual")

# Insert your Binance API key and secret
API_KEY = 'tTH25XYPnXjPQnOfTHbcd8y6FGaq6QXyIUf7jbR1iisyebqq5KByMyg8KFNHgn3h'
API_SECRET = 'fgPIxiygL5QvE5cPmopmCemxUYKQz2ThrAzFMEXz7nlPyeMcMaYnSrCHsyq62dAL'

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

simbolos_spot = Obtener_Simbolos_Binance(client)
st.write(f"Símbolos disponibles en Binance Spot: {len(simbolos_spot)}")
# Sidebar
with st.sidebar:
    col1, col2 = st.columns(2)
    st.title("📊 Parámetros de Análisis")
    
    symbols = ["NQ=F", "GOOG", "BTC-USD", "CNY=X", "TSLA", "EURUSD=X", "GC=F", "META", "NVDA",
               "BTCUSDT", "PEPEUSDT", "BLZUSDT", "1000PEPEUSDT", "ATAUSDT"] + simbolos_spot

    dict_exchange = {
        "binance": ["BTCUSDT", "PEPEUSDT", "BLZUSDT", "1000PEPEUSDT", "ATAUSDT"]+simbolos_spot,
        "yahoofinance": ["NQ=F", "GOOG", "BTC-USD", "CNY=X", "TSLA", "EURUSD=X", "GC=F", "META", "NVDA"]
    }

    simbol = st.selectbox("Símbolo 1", symbols, index=symbols.index("BTCUSDT"))
    
    interval_mayor = col1.selectbox("Intervalo H", ("1m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1W","1M"), index=10)
    interval_menor = col2.selectbox("Intervalo L", ("1m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1W","1M"), index=6)
    # Mostrar el resultado de la función
    # interval_mayor, interval_menor = diccionario_TimeFrames(interval)
    
    with col1:
        st.metric("TF Mayor", interval_mayor)

    with col2:
        st.metric("TF Menor", interval_menor)

    st.markdown("### Rango temporal de Análisis")
    periodos = st.number_input("Periodos a descargar", min_value=1, max_value=10000, value=5,
                               help="Cantidad de periodos que desea descargar hacia atrás.")
    periodicidad = st.selectbox("Periodicidad", ["minutos", "horas", "días", "semanas"], index=3,
                                help="Unidad de tiempo para los periodos hacia atrás.")

    
    excluir_final = st.number_input("Periodos a excluir", min_value=1, max_value=10000, value=7,
                               help="Cantidad de periodos que desea analizar hacia atrás.")
    rango_analisis = st.number_input("Periodos a analizar", min_value=1, max_value=10000, value=30,
                               help="Cantidad de periodos que desea analizar hacia atrás.")
    rango_analisis = (excluir_final, rango_analisis)
    
    # Convertimos los periodos y periodicidad a timedelta
    import datetime
    delta_dict = {
        "minutos": datetime.timedelta(minutes=periodos),
        "horas": datetime.timedelta(hours=periodos),
        "días": datetime.timedelta(days=periodos),
        "semanas": datetime.timedelta(weeks=periodos)
    }

    fecha_inicio = datetime.datetime.now() - delta_dict[periodicidad]
    fecha_inicio = fecha_inicio.strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"🕒 Fecha de inicio calculada: **{fecha_inicio}**")



# ✅ PARTE 2: Descarga Limpieza y procesamiento validación de datos
use_binance = 1 if simbol in dict_exchange["binance"] else 0
# Intentamos descargar los datos
df_mayor = preparar_dataset(client, simbol, interval_mayor, fecha_inicio, use_binance)
st.write(f"Columnas de dataset mayor: {df_mayor.columns}")
analizador = AnalizadorVelas(df_mayor, rango_analisis)
dict_grl_mayor = analizador.analizar()
# Luego de hacer el analaizar extraigo la información generada.
df_mayor = analizador.df_original
st.write(f"Columnas de dataset mayor: {df_mayor.columns}")

df_menor = preparar_dataset(client, simbol, interval_menor, fecha_inicio=df_mayor.shape[0], use_binance=use_binance)
analizador2 = AnalizadorVelas(df_menor, rango_analisis)

dict_grl_menor = analizador2.analizar()
# Luego de hacer el analaizar extraigo la información generada.
df_menor = analizador2.df_original

if df_mayor is None:
    st.warning(f"⚠️ No se pudo obtener o procesar el dataset para el timeframe mayor: {interval_mayor}")
    st.stop()

if df_menor is None:
    st.warning(f"⚠️ No se pudo obtener o procesar el dataset para el timeframe menor: {interval_menor}")
    st.stop()

# ✅ Parate 3: Mineria
df_mayor = detectar_swing_points(df_mayor)
df_menor = detectar_swing_points(df_menor)

# ✅ PARTE 4: Visualización e interfaz principal
st.title("📊 Análisis Técnico de Activos")
st.subheader(f"Símbolo seleccionado: `{simbol}` | Intervalo: `{interval_mayor} & {interval_menor}`")
st.success("Datos descargados y procesados exitosamente.")

parametros_indicadores = Solicita_Parametros_Indicadores()


# Mostrar qué seleccionó el usuario
if parametros_indicadores.keys():
    #print(indicadores_seleccionados)
    st.markdown(f"**Indicadores seleccionados:** {', '.join(parametros_indicadores.keys())}")
    
    #print(parametros_indicadores)
    df_mayor, cols_indicadores = Poner_Indicadores(df_mayor, parametros_indicadores)
    df_menor, _ = Poner_Indicadores(df_menor, parametros_indicadores)
    # Generamos los indicadores pertinentes
    analizador_mayor = AnalizadorIndicadores(df_mayor, parametros_indicadores)
    df_mayor = analizador_mayor.Genera_Indicadores()
    df_menor = AnalizadorIndicadores(df=df_menor,
                                            dict_parametros=parametros_indicadores).Genera_Indicadores()
    st.write(analizador_mayor.ult_value_indicator)
       
    col_medias = cols_indicadores["Media_Movil"]
else:
    col_medias = []
    st.markdown("⚠️ No se seleccionaron indicadores.")


tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "🗃 Datos", "📊 Estadísticas", "📉 Correlación"])


# Gráfico principal
with tab1:
    # Selección tipo de gráfica
    opcion_chart = st.selectbox('Tipo de gráfico', ['Vela', 'Línea',"Tamanio Vela"])
    # 📌 Análisis de rupturas de niveles para TimeFrame Mayor
    st.markdown(f"### ⏱️ TimeFrame Mayor: `{interval_mayor}` {df_mayor.shape[0]}registros") 
    # 📈 Gráfico mayor
    if opcion_chart == 'Vela':
        grafico = GraficoVelas(df_mayor, dict_grl_mayor, tab1)
        dict_fvg = dict_grl_mayor["FVGs"]
        print(dict_fvg.items() )
        fvg_no_tocadas = {
            k: v for k, v in dict_fvg.items()
            if isinstance(v, dict) and not v.get("tocada", True)
        }
        df_mayor = Contar_FVGs(df_mayor, fvg_no_tocadas)
        
        grafico.columnas_medias = col_medias
        grafico.graficar_velas(parametros_indicadores.keys())
        if "Estocastico" in parametros_indicadores.keys(): 
            graficar_estocastico(df_mayor)
    elif opcion_chart == 'Tamanio Vela':
        ventana = st.slider("Ventana de Tamanio Vela", min_value=1, max_value=100, value=20, step=1)
        grafica_tamanio_vela_interactiva(df_mayor["Tamanio_Vela"], window = ventana, title=f"Tamaño de Vela en {simbol} ({interval_mayor})")
        grafica_tamanio_vela_interactiva(df_menor["Tamanio_Vela"], window = ventana, title=f"Tamaño de Vela en {simbol} ({interval_menor})")
    else:
        fig_mayor = graficar_linea(df_mayor, ["Close"])
    
    # 📌 Análisis de rupturas de niveles para TimeFrame Menor
    st.markdown(f"### ⏱️ TimeFrame Menor: `{interval_menor}` {df_menor.shape[0]}registros")
    # 📉 Gráfico menor
    if opcion_chart == 'Vela':
        grafico = GraficoVelas(df_menor, dict_grl_menor, tab1)
        grafico.columnas_medias = col_medias
        grafico.graficar_velas(parametros_indicadores.keys())
        if "Estocastico" in parametros_indicadores.keys():
            graficar_estocastico(df_menor)
    else:
        fig_menor = graficar_linea(df_menor, ["Close"])


# Tabla
with tab2:
    features = ['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volumne', 'Quote_asset_vol', 'Number_trades', 
                'dia_num', 'mes', 'dia_mes', 'fecha', 'Volume', 'tipo', 'FVGs', 'Ruptura_Alcista', 
                'Tipo_Ruptura', 'Ruptura_Bajista', 'Tamanio_Vela', 'Tipo_Vela', 'Swing_Alcista', 'Swing_Bajista', 'EMA_20', 
                'SMA_200', 'CruceMM_EMA_20_SMA_200', 'DuracionTrendMM_EMA_20_SMA_200', 'Entrada_Trade', 'Tendencia_MM', 
                'Tendencia_PullBack', 'CruceMM_Close_SMA_200', 'DuracionTrendMM_Close_SMA_200', 'CruceMM_Close_EMA_20', 
                'DuracionTrendMM_Close_EMA_20', '%K', '%D', 'Estocastico_Senal', 'Duracion_Estocastico', 'FVGs_Falta', 'Duracion_FVGs_Falta']
    
    st.write(f"Columnas disponibles: {features}")
    n = st.number_input("N-registros", min_value=1, value=10, help="Cuantos renglones se imprimen")
    st.write(f"Últimas 10 filas de {simbol}")
   # st.write(df_mayor.columns)
    st.dataframe(df_mayor.tail(n).to_dict())
    
    # Entrenar el modelo
    trained_agent = train_ddpg_trader(df_mayor, features, episodes=50)
    
    # Generar señales
    df_mayor['signal'] = predict_signals(trained_agent, df_mayor)
    n = st.number_input("N-registros", min_value=1, value=10, help="Cuantos renglones se imprimen")
    st.write(f"Últimas 10 filas de {simbol}")
   # st.write(df_mayor.columns)
    st.dataframe(df_mayor.tail(n).to_dict())
    st.write(df_mayor.iloc[-1*n][["Close","Open"]].to_dict())

# Estadísticos
with tab3:
    last_close = round(df_mayor["Close"].iloc[-1], 5)
    #delta = round(df_mayor["Close_pc_1"].iloc[-2], 5)
    st.metric("📌 Precio de cierre", value=last_close)
    st.write(dict_grl_mayor)
    if parametros_indicadores.keys():
        st.write("")
    
    

    c1, c2 = st.columns(2)
    #c1.metric("📈 Media % cambio", round(df_mayor["Close_pc_1"].mean(), 5))
    #c2.metric("📉 Volatilidad", round(df_mayor["Close_pc_1"].std(), 5))
