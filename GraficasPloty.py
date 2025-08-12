import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta, datetime
import sys
import streamlit as st
from Std_streamlit import obtener_min_max #contar_salidas_de_limites, Cuenta_Fair_Value_Gap

# Añadir rutas personalizadas
sys.path.append('/home/hector/Documentos/Escuelas/Autodidacta/Git_repositories/Trading_test')

from Indicadores.Direccion import Medias_Moviles

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
    #fig.update_layout(title=title,
     #               xaxis_title='Tiempo',
      #              yaxis_title='Volumen transaccionado en miles.')
    #return fig


class GraficoVelas:
    def __init__(self, df, dict_general, obj_st):
        self.df = df
        self.dict_general = dict_general
        self.obj_st = obj_st
        self.fig = go.Figure()
        # Columnas de medias moviles
        self.columnas_medias = []

    def graficar_velas_base(self):
        self.fig.add_trace(go.Candlestick(
            x=self.df.index,
            open=self.df['Open'],
            high=self.df['High'],
            low=self.df['Low'],
            close=self.df['Close'],
            name="Velas"
        ))

    def graficar_rango(self):
        try:
            min_low, max_high = self.dict_general["Rango_Min_Max"]
            idx_min = self.df[self.df["Low"] == min_low].index[0]
            idx_max = self.df[self.df["High"] == max_high].index[0]

            self.fig.add_shape(
                type="line",
                x0=idx_min,
                y0=min_low,
                x1=self.df.index[-1],
                y1=min_low,
                line=dict(color="saddlebrown", width=1, dash="dash"),
                name="Mínimo"
            )

            self.fig.add_shape(
                type="line",
                x0=idx_max,
                y0=max_high,
                x1=self.df.index[-1],
                y1=max_high,
                line=dict(color="saddlebrown", width=1, dash="dash"),
                name="Máximo"
            )
        except Exception as e:
            self.obj_st.warning(f"No se pudo graficar el rango: {e}")

    def graficar_fvg(self):
        dict_fvg = self.dict_general.get("FVGs", {})
        for _, fvg_data in dict_fvg.items():
            try:
                if isinstance(fvg_data,dict):
                    # Convertir numpy.datetime64 a datetime
                    x0_np = fvg_data["fecha_inicio"]
                    x1_np = fvg_data["fecha_fin"]

                    x0 = pd.to_datetime(str(x0_np))
                    x1 = pd.to_datetime(str(x1_np))

                    y0, y1 = fvg_data["rango"]

                    self.fig.add_shape(
                        type="rect",
                        x0=x0,
                        x1=x1,
                        y0=y0,
                        y1=y1,
                        fillcolor="purple",
                        opacity=0.4,
                        line=dict(color="purple", width=1)
                        # Puedes agregar layer="below" después de que veas que funciona
                    )
                    #print(f"Se graficó FVG de {x0} a {x1}, rango {y0}-{y1}")
            except Exception as e:
                self.obj_st.warning(f"Error al graficar FVG: {e}")

    def mostrar_rupturas(self):
        conteo_rup = self.dict_general.get("Conteo_Rupturas", {})
        claves_visibles = {k: v for k, v in conteo_rup.items() if v > 0}

        if claves_visibles:
            self.obj_st.markdown("#### 📍 Rupturas recientes de niveles")
            cols = self.obj_st.columns(len(claves_visibles))
            for col, (clave, valor) in zip(cols, claves_visibles.items()):
                col.metric(label=clave, value=valor)
                
    def graficar_swings(self):
        try:
            if "Swing_Alcista" not in self.df.columns or "Swing_Bajista" not in self.df.columns:
                self.obj_st.warning("No se encontraron columnas de swing en el DataFrame.")
                return

            # Puntos swing alcistas (mínimos locales)
            df_alcistas = self.df[self.df["Swing_Alcista"] == 1]
            self.fig.add_trace(go.Scatter(
                x=df_alcistas.index,
                y=df_alcistas["Low"],
                mode="markers",
                marker=dict(symbol="triangle-up", color="green", size=10),
                name="Swing Alcista"
            ))

            # Puntos swing bajistas (máximos locales)
            df_bajistas = self.df[self.df["Swing_Bajista"] == 1]
            self.fig.add_trace(go.Scatter(
                x=df_bajistas.index,
                y=df_bajistas["High"],
                mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=10),
                name="Swing Bajista"
            ))
        except Exception as e:
            self.obj_st.warning(f"No se pudieron graficar los swings: {e}")
            
    def graficar_medias_moviles(self, columnas_medias=[]):
        """
        Grafica las medias móviles que se pasen por lista y marca los cruces si existen.
        """
        
        try:
            colores = ["orange", "lime", "magenta", "blue", "cyan", "yellow"]
            color_idx = 0

            # Dibujar las medias móviles
            for col in self.columnas_medias:
                if col in self.df.columns:
                    self.fig.add_trace(go.Scatter(
                        x=self.df.index,
                        y=self.df[col],
                        mode="lines",
                        line=dict(color=colores[color_idx % len(colores)], width=1),
                        name=col
                    ))
                    color_idx += 1
                else:
                    self.obj_st.warning(f"La columna '{col}' no existe en el DataFrame.")

            # Dibujar cruces (detecta columnas que empiecen con 'Cruce_')
            #cols_cruce = [c for c in self.df.columns if c.startswith("CruceMM_")]
            #for col_cruce in cols_cruce:
            #    df_cruce = self.df[self.df[col_cruce] != 0]
            #    for idx, row in df_cruce.iterrows():
            #        y = self.df.loc[idx, "Close"]
            #        signo = row[col_cruce]
            #        color = "white" if signo == 1 else "lightgrey"
#
            #        self.fig.add_trace(go.Scatter(
            #            x=[idx],
            #            y=[y],
            #            mode="markers",
            #            marker=dict(
            #                symbol="cross",
            #                size=12,
            #                color=color,
            #                line=dict(width=1, color="black")
            #            ),
            #            name=f"{col_cruce} ({'Alcista' if signo==1 else 'Bajista'})",
            #            showlegend=False
            #        ))
        except Exception as e:
            self.obj_st.warning(f"Error al graficar medias móviles: {e}")

    def graficar_velas(self, indicadores):
        self.mostrar_rupturas()
        self.graficar_velas_base()
        self.graficar_rango()
        self.graficar_fvg()
        self.graficar_swings() 
        
        if "Media_Movil" in indicadores:
            self.graficar_medias_moviles()

        self.fig.update_layout(
            title='📊 Gráfico de Velas con FVGs y Rango',
            xaxis_title='Fecha',
            yaxis_title='Precio',
            xaxis_rangeslider_visible=False
        )

        self.obj_st.plotly_chart(self.fig, use_container_width=True)
        return self.fig

import plotly.graph_objects as go
import streamlit as st

def graficar_estocastico(df, titulo="Indicador Estocástico"):
    """
    Grafica el Estocástico (%K y %D) con zonas de sobrecompra y sobreventa.
    
    Args:
        df (pd.DataFrame): Debe contener columnas '%K' y '%D' y el índice con la fecha.
        titulo (str): Título opcional del gráfico.
    """

    fig = go.Figure()

    # Línea %K
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['%K'],
        mode='lines',
        name='%K',
        line=dict(color='blue')
    ))

    # Línea %D
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['%D'],
        mode='lines',
        name='%D',
        line=dict(color='orange')
    ))

    # Zonas de sobrecompra y sobreventa
    fig.add_shape(
        type="rect",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=80,
        y1=100,
        fillcolor="purple",
        opacity=0.1,
        layer="below",
        line_width=0
    )

    fig.add_shape(
        type="rect",
        x0=df.index.min(),
        x1=df.index.max(),
        y0=0,
        y1=20,
        fillcolor="purple",
        opacity=0.1,
        layer="below",
        line_width=0
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="Fecha",
        yaxis_title="Valor Estocástico",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


    
def graficar_linea(data,obj_st, col_use, ytitle="Precio",title='Gráfico de Línea',fig=None, return_fig=False):
    # Crear la figura del histograma
    aux = False
    if fig==None:
        aux = True
        fig = go.Figure()
    if isinstance(col_use,str):
        col_use = [col_use]
    
    for col in col_use:
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='lines', name=col))
    if aux:
        fig.update_layout(title=title, xaxis_title='Fecha', yaxis_title=ytitle)
    if return_fig:
        return fig
    else: obj_st.plotly_chart(fig)
    
def graficar_histograma(data,col_use, obj_st, title="Histograma Rendimientos al cierre"):
    # Crear la figura del histograma
    fig = go.Figure()

    # Agregar el histograma a la figura
    fig.add_trace(go.Histogram(x=data[col_use], name=col_use))

    # Establecer el diseño del gráfico
    fig.update_layout(title=title,
                    xaxis_title=f'Valor {col_use}',
                    yaxis_title='Frecuencia')
    obj_st.plotly_chart(fig)
    
def graficar_burbujas(data,col_use, obj_st,title="Burbuja inflacionaria"):
    df_ordenado = data.sort_values(by=col_use, ascending=False)
    fig = go.Figure(data=[go.Scatter(
        x=df_ordenado["symbol"], y=df_ordenado[col_use],
        mode='markers',
        marker_size=df_ordenado[col_use])
    ])
    fig.update_layout(title=title,
                    xaxis_title=f'Valor {col_use}',
                    yaxis_title='Frecuencia')
    obj_st.plotly_chart(fig)
    
def graficar_scatter(df1,df2, cols:tuple, obj_st, title="Dispercion"):
    if df1.shape[0] == df2.shape[0] and len(cols)==2:
        fig = go.Figure(data=[go.Scatter(
            x=df1[cols[0]], y=df2[cols[1]],
            mode='markers',)
        ])
        fig.update_layout(title=title,
                        xaxis_title=f'Valor {cols[0]}',
                        yaxis_title=f"Valor {cols[1]}")
        obj_st.plotly_chart(fig)

def plot_ichimoku_cloud0(df, obj_st):
    """
    Grafica el Ichimoku Cloud sobre el DataFrame dado.
    
    Parámetros:
    df (DataFrame): DataFrame que contiene las columnas del Ichimoku Cloud.
    """

    plt.figure(figsize=(14, 8))

    plt.plot(df.index, df['Close'], label='Precio de Cierre', color='black', linewidth=2)
    plt.plot(df.index, df['Tenkan_sen'], label='Tenkan-sen', color='red', linewidth=1.5)
    plt.plot(df.index, df['Kijun_sen'], label='Kijun-sen', color='blue', linewidth=1.5)
    plt.plot(df.index, df['Senkou_Span_A'], label='Senkou Span A', color='green', linestyle='--')
    plt.plot(df.index, df['Senkou_Span_B'], label='Senkou Span B', color='brown', linestyle='--')
    plt.plot(df.index, df['Chikou_Span'], label='Chikou Span', color='purple', linewidth=1.5)

    # Relleno de la nube entre Span A y Span B
    plt.fill_between(df.index, df['Senkou_Span_A'], df['Senkou_Span_B'], where=(df['Senkou_Span_A'] >= df['Senkou_Span_B']), color='lightgreen', alpha=0.5)
    plt.fill_between(df.index, df['Senkou_Span_A'], df['Senkou_Span_B'], where=(df['Senkou_Span_A'] < df['Senkou_Span_B']), color='lightcoral', alpha=0.5)

    plt.title('Indicador Ichimoku Cloud')
    plt.legend(loc='upper left')
    plt.grid(True)
    # Mostrar el gráfico en Streamlit
    obj_st.pyplot(plt.gcf())
    
def plot_ichimoku_cloud(df,obj_st, title="Dispercion"):
    """
    Grafica el Ichimoku Cloud sobre el DataFrame dado.
    
    Parámetros:
    df (DataFrame): DataFrame que contiene las columnas del Ichimoku Cloud.
    """

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Precio de Cierre', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan_sen'], mode='lines', name='Tenkan-sen', line=dict(color='red', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Kijun_sen'], mode='lines', name='Kijun-sen', line=dict(color='blue', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_Span_A'], mode='lines', name='Senkou Span A', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_Span_B'], mode='lines', name='Senkou Span B', line=dict(color='brown', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Chikou_Span'], mode='lines', name='Chikou Span', line=dict(color='purple', width=1.5)))

      # Relleno de la nube entre Span A y Span B
    fig.add_trace(go.Scatter(
        x=np.concatenate([df.index, df.index[::-1]]),
        y=np.concatenate([df['Senkou_Span_A'], df['Senkou_Span_B'][::-1]]),
        fillcolor='lightgreen',
        where=(df['Senkou_Span_A'] >= df['Senkou_Span_B'])
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([df.index, df.index[::-1]]),
        y=np.concatenate([df['Senkou_Span_B'], df['Senkou_Span_A'][::-1]]),
        fillcolor='lightcoral',
        where=(df['Senkou_Span_A'] < df['Senkou_Span_B'])
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Fecha',
        yaxis_title='Precio',
        legend_title='Elementos',
        template='plotly_white'
    )
    
    obj_st.plotly_chart(fig)
    return fig