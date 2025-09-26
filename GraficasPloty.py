import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta, datetime
import sys
import streamlit as st
from Std_streamlit import obtener_min_max #contar_salidas_de_limites, Cuenta_Fair_Value_Gap
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
    
    def graficar_trades(self):
        try:
            if "Entrada_Trade" not in self.df.columns:
                self.obj_st.warning("No se encontró la columna 'Entrada_Trade' en el DataFrame.")
                return

            # Filtrar entradas (1)
            df_entradas = self.df[self.df["Entrada_Trade"] == 1]
            self.fig.add_trace(go.Scatter(
                x=df_entradas.index,
                y=df_entradas["Close"],
                mode="markers",
                marker=dict(symbol="arrow-up", color="blue", size=12),
                name="Entrada"
            ))

            # Filtrar salidas (-1)
            df_salidas = self.df[self.df["Entrada_Trade"] == -1]
            self.fig.add_trace(go.Scatter(
                x=df_salidas.index,
                y=df_salidas["Close"],
                mode="markers",
                marker=dict(symbol="arrow-down", color="orange", size=12),
                name="Salida"
            ))

        except Exception as e:
            self.obj_st.warning(f"No se pudieron graficar las entradas/salidas: {e}")

            
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
        #self.graficar_swings() 
        #self.graficar_trades()
        
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
    
def graficar_burbujas(data,col_use, title="Burbuja inflacionaria"):
    df_ordenado = data.sort_values(by=col_use, ascending=False)
    fig = go.Figure(data=[go.Scatter(
        x=df_ordenado["Simbolo"], y=df_ordenado[col_use],
        mode='markers',
        marker_size=df_ordenado[col_use])
    ])
    fig.update_layout(title=title,
                    xaxis_title=f'Valor {col_use}',
                    yaxis_title='Frecuencia')
    st.plotly_chart(fig)
    
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


def grafica_tamanio_vela(serie: pd.Series, window: int = 20):
    """
    Visualiza en Streamlit el tamaño de las velas y su media móvil.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(serie.index, serie.values, label="Tamaño vela", alpha=0.6)
    ax.plot(serie.index, serie.rolling(window).mean(), 
            label=f"Media móvil {window}", color="red", linewidth=2)

    ax.set_title("Evolución del tamaño de las velas (volatilidad)")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Tamaño (High - Low)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    
import streamlit as st
import plotly.express as px
import pandas as pd

def grafica_tamanio_vela_interactiva(serie: pd.Series, window: int = 20,title= "Tamaño de Vela Interactiva"):
    """
    Visualiza en Streamlit el tamaño de las velas con Plotly interactivo.
    """
    df_temp = pd.DataFrame({
        "Fecha": serie.index,
        "Tamano_Vela": serie.values,
        "Media_Movil": serie.rolling(window).mean().values
    })

    fig = px.line(df_temp, x="Fecha", y=["Tamano_Vela", "Media_Movil"],
                  labels={"value": "Tamaño", "variable": "Serie"},
                  title=title)
    st.plotly_chart(fig, use_container_width=True)



import matplotlib.pyplot as plt
import streamlit as st

def plot_top_bottom_volumen(df, col_use, top_n=20):
    """
    Genera gráficos de Top N y Bottom N activos por Volumen.
    
    Parámetros:
        df (DataFrame): debe tener columnas ["Simbolo", "Volumen"]
        top_n (int): número de activos a mostrar en top y bottom
    """
    # Ordenamos por volumen
    df_sorted = df.sort_values(col_use, ascending=False)

    # Top N y Bottom N
    topN = df_sorted.head(top_n)
    bottomN = df_sorted.tail(top_n)

    figs = []

    # ----- Top N -----
    fig_top, ax_top = plt.subplots(figsize=(10,6))
    ax_top.barh(topN["Simbolo"], topN[col_use], color="green")
    ax_top.invert_yaxis()
    ax_top.set_title(f"Top {top_n} activos por {col_use}")
    ax_top.set_xlabel(col_use)
    ax_top.set_ylabel("Símbolo")
    figs.append(fig_top)

    # ----- Bottom N -----
    fig_bottom, ax_bottom = plt.subplots(figsize=(10,6))
    ax_bottom.barh(bottomN["Simbolo"], bottomN[col_use], color="red")
    ax_bottom.invert_yaxis()
    ax_bottom.set_title(f"Bottom {top_n} activos por Volumen")
    ax_bottom.set_xlabel(col_use)
    ax_bottom.set_ylabel("Símbolo")
    #figs.append(fig_bottom)

    return fig_top, topN["Simbolo"].values.tolist()


import plotly.graph_objects as go
import streamlit as st

def plot_volumen_heatmap(df, col_use, top_n=50):
    """
    Genera un heatmap de los activos por volumen.
    
    Parámetros:
        df (DataFrame): columnas ["Simbolo", "Volumen"]
        top_n (int): número de activos a mostrar
    """
    # Ordenamos por volumen (de mayor a menor)
    df_sorted = df.sort_values(col_use, ascending=False).head(top_n)

    # Heatmap (1 fila, cada columna es un símbolo)
    fig = go.Figure(data=go.Heatmap(
        z=[df_sorted[col_use].values],  
        x=df_sorted["Simbolo"].values,    
        y=[col_use],                    
        colorscale="Viridis",             # Escala de colores (puedes probar "RdYlGn" o "Plasma")
        text=df_sorted[col_use].values, # Tooltip con valor
        texttemplate="%{text}",           # Mostrar volumen sobre cada celda
        hovertemplate="Activo: %{x}<br>"+str(col_use)+": %{z}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Heatmap de {col_use} - Top {top_n} Activos",
        xaxis_title="Activo",
        yaxis_title="",
        xaxis=dict(showticklabels=True, tickangle=45),
        height=400
    )

    return fig, df_sorted["Simbolo"].values.tolist()



def plot_volumen_bubbles(df, col_use="Volumne", top_n=50):
    """
    Genera un gráfico de burbujas flotantes para los activos.
    
    Parámetros:
        df (DataFrame): columnas ["Simbolo", col_use]
        col_use (str): columna de volumen a usar ("Volumne" o "Volumen_acum")
        top_n (int): número de activos a mostrar
    """
    # Ordenamos y seleccionamos los más relevantes
    df_sorted = df.sort_values(col_use, ascending=False).head(top_n)
    st.write(f"Hay {len(df_sorted)} activos para graficar.")

    # Posiciones aleatorias para dispersar las burbujas
    np.random.seed(42)  # fijo para reproducibilidad
    x = np.random.uniform(0, 1, len(df_sorted))
    y = np.random.uniform(0, 1, len(df_sorted))

    # Escalamos el tamaño de las burbujas
    sizes = df_sorted[col_use] / df_sorted[col_use].max() * 2000  # factor ajustable

    # Texto dentro de la burbuja (símbolo + volumen)
    labels = [f"{s}<br>{v:,}" for s, v in zip(df_sorted["Simbolo"], df_sorted[col_use])]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=labels,
        textposition="middle center",
        marker=dict(
            size=sizes,
            color=df_sorted[col_use],
            colorscale="Viridis",
            showscale=True,
            sizemode="area",
            line=dict(width=2, color="DarkSlateGrey")
        ),
        hovertemplate="Activo: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Gráfico de Burbujas - Top {top_n} Activos ({col_use})",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=700
    )

    return fig

