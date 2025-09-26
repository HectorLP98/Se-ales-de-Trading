from Indicadores.Volatilidad import *
from Indicadores.Volumen import *
from Indicadores.Direccion import *
from Indicadores.Momento import *
from Indicadores.Trend import *
from Indicadores.Optimizacion import *


class Estrategias:
    long_open_sings = []
    long_close_signs = []
    short_open_signs = []
    short_close_sings = []
    rows = []
    operaciones = {"Abiertas":0,"Cerradas":0,"Ganadoras":0,"Perdedoras":0}
    tipo_entrada = None
    precio_entrada = None
    show_print = False
    mm = Medias_Moviles()
    
    def __init__(self) -> None:
        operaciones = {"Abiertas":0,"Cerradas":0,"Ganadoras":0,"Perdedoras":0}
        pass
    
    def CountTrend(self, serie):
        """
        (Function)
            Esta funcion cuenta el numero de periodos que duro un Trend al timeframe pasado.
            Por lo que returna una serie secuencial de 0 a n, donde n determina la cantidad de periodos que duro el Trend.
        (Parameters)
            serie[iterator] : Datos de los cuales se parten.
        """
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
    
    def reiniciar_variables(self):
        self.tipo_entrada = None
        self.rows = []
        self.long_open_sings = []
        self.long_close_signs = []
        self.short_open_signs = []
        self.short_close_sings = []
        self.operaciones = {"Abiertas":0,"Cerradas":0, "Ganadoras":0,"Perdedoras":0}
        
    def calcular_pseudorendimiento(self,capital,i):
        if self.tipo_entrada=="long":
            # Cerrar posición larga
            self.cerrar_posicion("long", capital, i, psuedo_cierre=True)
        elif self.tipo_entrada=="short":
            # Cerrar posición corta
            self.cerrar_posicion("short", capital, i, psuedo_cierre=True)
        else:
            self.rows.append((capital,self.df.index[i],"",self.df.Close[i],0))
    
    def abrir_posicion(self, tipo,capital,i):
        if tipo=="long":
            # Abrir posición larga
            if self.show_print: print(f"Entro apertura de long: Precio: {self.df['Close'].iloc[i]},    Fecha: {self.df.index[i]}")
            self.precio_entrada = self.df['Close'].iloc[i]
            self.rows.append((capital, self.df.index[i], 'Apertura Long',self.df.Close[i],0))
            self.operaciones["Abiertas"] +=1
            self.tipo_entrada = "long"
            self.long_open_sings.append((self.df.index[i],self.df["Close"][i]))
            
        elif tipo=="short":
            # Abrir posición corta
            if self.show_print: print(f"Entro apertura de short: Precio: {self.df['Close'].iloc[i]},    Fecha: {self.df.index[i]}")
            self.precio_entrada = self.df['Close'].iloc[i]
            self.tipo_entrada = "short"
            self.rows.append((capital, self.df.index[i], 'Apertura Short',self.df.Close[i], 0))
            self.operaciones["Abiertas"] +=1
            self.short_open_signs.append((self.df.index[i],self.df["Close"][i]))
        return capital
            
    def cerrar_posicion(self, tipo, capital, i, psuedo_cierre=False):
        if tipo=="long":
            # Cerrar posición larga
            if psuedo_cierre==False:
                rendimiento = (self.df['Close'].iloc[i] - self.precio_entrada) / self.precio_entrada
                #print( self.precio_entrada, self.df['Close'][i],rendimiento )
                capital *= (1 + rendimiento)
                capital *= (1 - self.comision)  # Aplicar comisión al cerrar la posición
                if self.show_print: print(f"Entro cierre de long: Precio: {self.df['Close'].iloc[i]},    Fecha: {self.df.index[i]},    Capital: {capital}")
                if (self.df['Close'].iloc[i] - self.precio_entrada) > 0: flg_winner = 1 
                else: flg_winner = -1
                self.rows.append((capital, self.df.index[i], 'Cierre Long',self.df.Close[i],flg_winner))
                self.tipo_entrada = None
                self.precio_entrada = None
                self.operaciones["Cerradas"] +=1
                if flg_winner==1: self.operaciones["Ganadoras"] +=1
                else: self.operaciones["Perdedoras"] +=1
                self.long_close_signs.append((self.df.index[i],self.df["Close"][i]))
                return capital
            else:
                rendimiento = (self.df['Close'][i] - self.precio_entrada) / self.precio_entrada
                capital_aux = capital * (1 + rendimiento)
                capital_aux *= (1 - self.comision)  # Aplicar comisión al cerrar la posición
                self.rows.append((capital_aux, self.df.index[i], '',self.df.Close[i],0))
            
            
        elif tipo=="short":
            # Cerrar posición corta
            rendimiento = (self.precio_entrada - self.df['Close'].iloc[i]) / self.precio_entrada
            
            if not psuedo_cierre:
                #print( self.precio_entrada, self.df['Close'][i],rendimiento )
               
                capital *= (1 + rendimiento)
                capital *= (1 - self.comision)  # Aplicar comisión al cerrar la posición
                if self.precio_entrada - self.df['Close'].iloc[i] > 0: flg_winner = 1
                else: flg_winner = -1
                if self.show_print: print(f"Entro cierre de short: Precio: {self.df['Close'].iloc[i]},    Fecha: {self.df.index[i]},    Capital: {capital}")
                self.rows.append((capital, self.df.index[i], 'Cierre Short',self.df.Close[i],flg_winner))
                self.tipo_entrada = None
                self.precio_entrada = None
                self.operaciones["Cerradas"] +=1
                if flg_winner==1: self.operaciones["Ganadoras"] +=1
                else: self.operaciones["Perdedoras"] +=1
                self.short_close_sings.append((self.df.index[i],self.df["Close"][i]))
                return capital
            else:
                capital_aux = capital * (1 + rendimiento)
                capital_aux *= (1 - self.comision)  # Aplicar comisión al cerrar la posición
                self.rows.append((capital_aux, self.df.index[i], '',self.df.Close[i],0))
                
    def Señal_rsi(self, i,col_RSI="RSI", rsi_zones=[20,80]):
        try:
            if self.df[col_RSI][i] >rsi_zones[0] : #and self.df[col_RSI][i-1] <= rsi_zones[0] :
                return "sobreventa"
            elif self.df[col_RSI][i] < rsi_zones[1] : #and self.df[col_RSI][i-1] >=rsi_zones[1]:
                return "sobrecompra"
        except Exception as e:
            raise ValueError(f"----- No se genero el {col_RSI}: {e}")
    
    def Señal_MM(self, i, col_MM):
        if col_MM in ("EMA","SMA","WMA"):
            col_0 = col_MM+"0"
            col_1 = col_MM+"1"
        else:
            col_0 = col_1 = col_MM
    
        if self.df["Cross"+col_MM][i] >0:
            return "cruce_alcista"
        elif self.df["Cross"+col_MM][i] <0:
            return "cruce_bajista"
        elif self.df[col_0].iloc[i] > self.df[col_1].iloc[i]:
            return "alcista"
        elif self.df[col_0].iloc[i] < self.df[col_1].iloc[i]:
            return "bajista"
        
    
    def Put_indicators(self,indicadores:dict):
        """_summary_
        Esta funcion coloca los indicadores del input, por ende este es un diccionario, ej {"EMA":[7,30],"SVO":20, "RSI":14}

        Args:
            indicator (dict): ej {"EMA":[7,30],"SVO":20, "RSI":14,"MACD":([14,28],9)}
        """
        mm = Medias_Moviles()
        # Estocastico
        if sum([1 if i.lower() in ("estocastico","stochastic","todos","all") else 0 for i in indicadores.keys()]) >0:
            key = [i  for i in indicadores.keys() if i.lower() in ("estocastico","stochastic")] 
            try:
                n, d = indicadores[key[0]]
            except:
                raise NameError("El indicador estocastico debe llamarse asi; {'estocastico':[n,d]} donde se calcula sobre n-periodos y se suaviza a d-periodos")

            self.df = calcular_estocastico(self.df, n, d)
            
        # MACD
        if sum([1 if i.upper() in ("MACD") else 0 for i in indicadores.keys()]) >0:
            key = [i  for i in indicadores.keys() if i.upper() in ("MACD","TODOS","ALL")] 
            try:
                periodos_macd, period_señal = indicadores[key[0]]
            except:
                raise NameError("El indicador MACD debe llamarse asi; {'MACD':[periodos_macd:list, period_señal:int]} donde se calcula sobre n-periodos y se suaviza a d-periodos")

            self.df = mm.calcular_MACD(self.df.copy(),col_use="Close",fastLength=periodos_macd[0], slowLength=periodos_macd[1],signalLength=period_señal)
            self.df["MACDTrend"] = self.CountTrend(self.df["Histograma"])
            
        # SVO
        if sum([1 if i.upper() in ("SVO","TODOS","ALL") else 0 for i in indicadores.keys()]) >0:
            key = [i  for i in indicadores.keys() if i.upper() in ("SVO","TODOS","ALL")] 
            try:
                period_SVO = indicadores[key[0]]
            except:
                raise NameError("El indicador SVO debe llamarse asi; {'SVO': n} donde se calcula sobre n-periodos")
            
            self.df["SVO"] = simple_volume_oscillator(self.df["Volumne"], period_SVO)
        
        # Bandas Bollinger
        if sum([1 if i.upper() in ("BB","BANDAS_BOLLINGER","TODOS","ALL") else 0 for i in indicadores.keys()]) >0:
            key = [i  for i in indicadores.keys() if i.upper() in ("BB","BANDAS_BOLLINGER","TODOS","ALL")] 
            try:
                periodos_bb, std_bb = indicadores[key[0]]
            except:
                raise NameError("El indicador Bandas Bollinger debe llamarse asi; {'BB': [periodos_bb:int, std_bb:float]} donde se calcula sobre los periodos_bb con una desviacion de std_bb")
            
            self.df = calcular_bandas_bollinger(self.df,ventana=periodos_bb,num_std=std_bb)
            self.df["distancia_BB_SUP"] = self.df["Banda_Superior"] - self.df["Close"]
            self.df["distancia_BB_INF"] = self.df["Close"] - self.df["Banda_Inferior"]
            self.df["BBTrend_SUP"] = self.CountTrend(self.df["distancia_BB_SUP"])
            self.df["BBTrend_INF"] = self.CountTrend(self.df["distancia_BB_INF"])
            
        # Medias moviles
        if sum([1 if i.upper() in ("MM","MEDIAS_MOVILES","TODOS","ALL") else 0 for i in indicadores.keys()]) >0:
            key = [i  for i in indicadores.keys() if i.upper() in ("MM","MEDIAS_MOVILES","TODOS","ALL")] 
            try:
                periodos_mm = indicadores[key[0]]
            except:
                raise NameError("El indicador Medias moviles debe llamarse asi; {'MM': periodos_mm:list} donde se calcula sobre los periodos_mm los cuales son [media_corta, media_larga] ")
            
            self.df = mm.Colocar_Medias_Moviles(df1=self.df,columnas={"precio":"Close","volumen":"Volumne"}, periodos_mm=periodos_mm)
            
        # RSI
        if sum([1 if i.upper() in ("RSI","TODOS","ALL") else 0 for i in indicadores.keys()]) >0:
            key = [i  for i in indicadores.keys() if i.upper() in ("RSI","TODOS","ALL")] 
            try:
                period_RSI = indicadores[key[0]]
            except:
                raise NameError("El indicador Medias moviles debe llamarse asi; {'MM': periodos_mm:list} donde se calcula sobre los periodos_mm los cuales son [media_corta, media_larga] ")
            
            self.df["RSI"] = calcular_rsi(self.df["Close"], periodos=period_RSI)
          
                
    def Plot_senales(self,ax1,periodos):
        
        long_open_sings = np.array(self.long_open_sings)
        long_close_sings = np.array(self.long_close_signs)
        short_open_signs = np.array(self.short_open_signs)
        short_close_signs = np.array(self.short_close_sings)  
        # verde bandera: #008000  rojo fuerte: #B22222
        if len(long_open_sings) > 0:
            ax1.scatter(long_open_sings[:,0], long_open_sings[:,1], color='#008000', s=100, zorder=5, label=f'Abré long({periodos})')
        if len(long_close_sings) >0:
            ax1.scatter(long_close_sings[:,0], long_close_sings[:,1], color='#B22222', s=100, zorder=5, label=f'Cierre long({periodos})')
        if len(short_open_signs) > 0:
            ax1.scatter(short_open_signs[:,0], short_open_signs[:,1], color='lightgreen', s=100, zorder=5, label=f'Abré short({periodos})')
        if len(short_close_signs) > 0:
            ax1.scatter(short_close_signs[:,0], short_close_signs[:,1], color='lightcoral', s=100, zorder=5, label=f'Cierre short({periodos})')
        
        ax1.set_xlabel('Fecha')
        ax1.set_ylabel('Precio')
        ax1.legend()
        ax1.grid(True)
            
    def Plot_Cartera(self,ax):
        # Convertir la lista de historia de capital a un DataFrame para facilitar el análisis
        capital_history_df = pd.DataFrame(self.rows, columns=['Capital', 'Fecha', 'Tipo',"Close","Ganadora"])
        
        # Generar la gráfica del valor de la cartera a lo largo del tiempo
        ax.plot(capital_history_df['Fecha'], capital_history_df['Capital'], marker='', linestyle='-', color='b')
        for index, row in capital_history_df.iterrows():
            if row["Tipo"]:
                color = 'green' if 'Apertura' in row['Tipo'] else 'red'
                ax.scatter(row['Fecha'], row['Capital'], color=color, s=100, zorder=5)
                if color =="green":
                    ax.text(row['Fecha'], row['Capital']*(1.008), str(row['Tipo']).split(" ")[1], fontsize=9, color=color, ha='left', va='bottom')
                else:
                    ax.text(row['Fecha'], row['Capital']*(1.005), str(row['Tipo']).split(" ")[1], fontsize=9, color=color, ha='left', va='bottom')
                
        # Añadir texto con la información del diccionario de manera independiente en la gráfica
        ax.text(0.95, 0.95, f"Abiertas: {self.operaciones['Abiertas']}\nCerradas: {self.operaciones['Cerradas']}\nGanadoras: {self.operaciones['Ganadoras']}\nPerdedoras: {self.operaciones['Perdedoras']}", 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title('Valor de la Cartera a lo Largo del Tiempo')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor de la Cartera ($)')
        ax.grid(True)
        ax.legend()
        print(f"Operaciones ganadoras: {capital_history_df[capital_history_df['Ganadora']==1]['Ganadora'].sum()}")
        self.df_cartera = capital_history_df.copy()
        
    def Plot_Estrategia_MACD(self, df, axes, symbol, period_señal=9, periodos_macd=[7,30], period_SVO=20,
                            capital_inicial=100, comision=0.001, show_print=False, put_SVO=False,  
                            put_plot=True):
        # Generamos el MACD y SVO
        self.df = df.copy()
        self.Put_indicators({"SVO":period_SVO,"MACD":(periodos_macd,period_señal)})
        
        if put_plot:
            try:
                if put_SVO:
                    ax1, ax2, ax3, ax4 = axes
                    ax3.plot(self.df.index, self.df['SVO'], label=f'SVO({period_SVO})', color='purple')
                    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
                    ax3.legend()
                    ax3.grid(True)
                else:
                    ax1, ax2, ax3 = axes
                ax1.plot(self.df.index, self.df['Close'], label='Close', color='black')
                ax2.plot(self.df.index, self.df['MACD'], label='MACD', color='blue')
                ax2.plot(self.df.index, self.df['Signal_Line'], label='Signal', color='orange')
                ax1.legend()
                ax1.grid(True)
                ax2.legend()
                ax2.grid(True)
            except:
                raise IndexError("Solo debe haber dos ejes para Estrategia_MACD")
            
        # Inicializamos variables
        capital = capital_inicial
        self.show_print = show_print
        self.comision = comision
        self.reiniciar_variables()
        
        # Determinar las señales de compra y venta basadas en las BB
        for i in range(1, len(self.df)):
            # Estrategia compra long: 
            if self.Señal_MM(i,col_MM="MACD")=="cruce_alcista" and self.tipo_entrada==None and self.df["MACD"][i] <0:
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.abrir_posicion("long",capital,i)
                elif put_SVO==False :
                    capital = self.abrir_posicion("long",capital,i)
                continue
            # Estrategia compra short
            elif self.Señal_MM(i,col_MM="MACD")=="cruce_bajista" and self.tipo_entrada==None and self.df["MACD"][i] >0:
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.abrir_posicion("short",capital,i)
                elif put_SVO==False :
                    capital = self.abrir_posicion("short",capital,i)
                continue
            # Estrategia cierre long
            elif self.Señal_MM(i,col_MM="MACD")=="cruce_bajista" and self.tipo_entrada=="long":
                if put_SVO and self.df["SVO"][i] <0:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
                elif put_SVO==False:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
            # Estrategia cierre short
            elif self.Señal_MM(i,col_MM="MACD")=="cruce_alcista" and self.tipo_entrada=="short":
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
                elif put_SVO==False:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
            
            # Calcular el pseudo - rendimiento
            self.calcular_pseudorendimiento(capital,i)
        
        # Ploteamos las señales
        if put_plot:
            self.Plot_senales(ax1,periodos_macd)
            ax1.set_title(f'{symbol} con SVO')
            
        if show_print:
            print(f"Capital final: {capital}")
        
        if put_SVO and put_plot:
            self.Plot_Cartera(ax=ax4)
        elif put_plot:
            self.Plot_Cartera(ax=ax3)
        return capital, periodos_macd, period_señal, period_SVO, self.operaciones, self.df["MACDTrend"].iloc[-1]
        
                
    def Plot_Estrategia_BB(self, df, axes, symbol, periodos_bb=20, std_bb=2,
                           tipo="mean_reversion", capital_inicial=100, comision=0.001, 
                           show_print=False, put_plot=False):
        ax1, ax2 = axes
        self.df = df.copy()
        # Generamos las BB
        self.Put_indicators({"BB":(periodos_bb, std_bb)})
        
        
        
        # Plotemaos las bandas B
        if put_plot:
            ax1.plot(self.df.index, self.df['Close'], label='Close', color='black')
            ax1.plot(self.df.index, self.df[f'Banda_Media'], label=f'Banda_Media({periodos_bb})', color='b')
            ax1.plot(self.df.index, self.df[f'Banda_Superior'], label=f'Banda_Superior({periodos_bb})', color='red')
            ax1.plot(self.df.index, self.df[f'Banda_Inferior'], label=f'Banda_Inferior({periodos_bb})', color='red')
        
        # Inicializamos variables
        capital = capital_inicial
        self.show_print = show_print
        self.comision = comision
        self.reiniciar_variables()

        # Determinar las señales de compra y venta basadas en las BB
        for i in range(1, len(self.df)):
            # Estrategia de Compra: Precio toca o rompe la Banda Inferior y vuelve a entrar
            if self.df["Close"][i-1] <= self.df["Banda_Inferior"][i-1] and self.df["Close"][i] >= self.df["Banda_Inferior"][i] and self.tipo_entrada==None and tipo=="mean_reversion":
                capital = self.abrir_posicion("long",capital,i)
                continue
            
            # Estrategia de Venta: Precio toca o rompe la Banda Inferior
            if self.df["Close"][i] < self.df["Banda_Inferior"][i] and self.df["Close"][i-1] >= self.df["Banda_Inferior"][i-1] and self.tipo_entrada==None and tipo=="breakout":
                capital = self.abrir_posicion("short",capital,i)
                continue
            
            # Estrategia de Venta: Precio toca o rompe la Banda Superior y vuelve a entrar
            elif self.df["Close"][i-1] >= self.df["Banda_Superior"][i-1] and self.df["Close"][i] <= self.df["Banda_Superior"][i] and self.tipo_entrada==None and tipo=="mean_reversion":
                capital = self.abrir_posicion("short",capital,i)
                continue
            
            # Estrategia de Venta: Precio toca o rompe la Banda Superior
            elif self.df["Close"][i] > self.df["Banda_Superior"][i] and self.df["Close"][i-1] <= self.df["Banda_Superior"][i-1] and self.tipo_entrada==None and tipo=="breakout":
                capital = self.abrir_posicion("long",capital,i)
                continue
                    
            # Estrategia de Salida: Cerrar long posición cuando el precio alcanza la Banda Media
            elif self.tipo_entrada=="long" and self.df["Close"][i] >= self.df["Banda_Media"][i] and tipo in ("mean_reversion","breakout"):
                capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                continue
            elif self.tipo_entrada=="short" and self.df["Close"][i] <= self.df["Banda_Media"][i] and tipo in ("mean_reversion","breakout"):
                capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                continue
            
            # Estrategia de Salida: Cuando el precio se mueve en contra
            
                
            # Calcular el pseudo - rendimiento
            elif self.tipo_entrada=="long":
                # Cerrar posición larga
                self.cerrar_posicion("long", capital, i, psuedo_cierre=True)
                continue
            elif self.tipo_entrada=="short":
                # Cerrar posición corta
                self.cerrar_posicion("short", capital, i, psuedo_cierre=True)
                continue
            self.rows.append((capital,self.df.index[i],"",self.df.Close[i],0))
            
            
        # Ploteamos las señales
        if put_plot:
            self.Plot_senales(ax1,periodos_bb)
            ax1.set_title(f'{symbol} con Bandas Bollinger')
            ax1.set_xlabel('Fecha')
            ax1.set_ylabel('Precio')
            ax1.legend()
            ax1.grid(True)
            print(f"Capital final: {capital}")
            self.Plot_Cartera(ax=ax2)
        return capital, periodos_bb, std_bb, self.operaciones, self.df["BBTrend_SUP"].iloc[-1], self.df["BBTrend_INF"].iloc[-1]
        
    def Plot_Estrategia_MM_SVO(self, df, axes, symbol, col_MM="EMA", periodos_mm=[7, 30], period_SVO=20, 
                               put_SVO=True, capital_inicial=100, comision=0.001, show_print=False,
                               salida_SVO=True, put_plot=False):
        
        # Generamos las medias moviles y SVO
        self.df = df.copy()
        self.Put_indicators({"MM":periodos_mm, "SVO":period_SVO})
        
        if put_SVO and put_plot:
            try:
                ax1, ax2, ax3 = axes
                ax2.plot(self.df.index, self.df['SVO'], label=f'SVO({period_SVO})', color='purple')
                ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
                ax2.legend()
                ax2.grid(True)
            except:
                raise ValueError("Debe haber 3 ejes para plotear")
        else:
            try:
                if put_plot:
                    ax1, ax2 = axes
            except:
                raise ValueError("Debe haber 2 ejes para plotear")
               
        # Ploteamos medias y precio cierre.
        if put_plot:
            ax1.plot(self.df.index, self.df['Close'], label='Close', color='black')
            ax1.plot(self.df.index, self.df[f'{col_MM}_{periodos_mm[0]}'], label=f'{col_MM}{periodos_mm[0]}', color='blue')
            ax1.plot(self.df.index, self.df[f'{col_MM}_{periodos_mm[1]}'], label=f'{col_MM}{periodos_mm[1]}', color='orange')
        
        # Inicializamos variables
        capital = capital_inicial
        self.show_print = show_print
        self.comision = comision
        self.reiniciar_variables()
        
        # Determinar las señales de compra y venta basadas en las BB
        for i in range(1, len(self.df)):
            # Estrategia de long: 
            if self.df["Cross"+col_MM][i] >0 and self.df["SVO"][i] >0 and self.tipo_entrada==None:
                capital = self.abrir_posicion("long",capital,i)
                continue
            # Estrategia de short: 
            elif self.df["Cross"+col_MM][i] <0 and self.df["SVO"][i] <0 and self.tipo_entrada==None:
                capital = self.abrir_posicion("short",capital,i)
                continue
            # Estrategia de Salida long:
            elif self.tipo_entrada=="long" and self.df["Cross"+col_MM][i] <0:
                if salida_SVO and self.df["SVO"][i] <0:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
                else:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
            # Estrategia de Salida short:
            elif self.tipo_entrada=="short" and self.df["Cross"+col_MM][i] >0:
                if salida_SVO and self.df["SVO"][i] >0:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
                else:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
            # Calcular el pseudo - rendimiento
            elif self.tipo_entrada=="long":
                # Cerrar posición larga
                self.cerrar_posicion("long", capital, i, psuedo_cierre=True)
                continue
            elif self.tipo_entrada=="short":
                # Cerrar posición corta
                self.cerrar_posicion("short", capital, i, psuedo_cierre=True)
                continue
            self.rows.append((capital,self.df.index[i],"",self.df.Close[i],0))
            
        # Ploteamos las señales
        if put_plot:
            self.Plot_senales(ax1,periodos_mm)
            ax1.set_title(f'{symbol} con SVO')
            ax1.set_xlabel('Fecha')
            ax1.set_ylabel('Precio')
            ax1.legend()
            ax1.grid(True)
            print(f"Capital final: {capital}")
        
        if put_SVO and put_plot:
            self.Plot_Cartera(ax=ax3)
        else:
            if put_plot:
                self.Plot_Cartera(ax=ax2)
        return capital, periodos_mm, period_SVO, self.operaciones
    
    def Plot_Estrategia_RSI(self, df, axes, symbol, period_RSI=12, period_SVO=20, periodos_mm=[7,30],
                            capital_inicial=100, comision=0.001, show_print=False, put_SVO=False,  
                            rsi_zones=[20,80], put_plot=True, col_MM="EMA"):
        
        # Generamos el RSi
        self.df = df.copy()
        self.Put_indicators({"MM":periodos_mm,"SVO":period_SVO, "RSI":period_RSI})
        
        if put_SVO:
            try:
                if put_plot:
                    ax1, ax2, ax3, ax4 = axes
                    ax3.plot(self.df.index, self.df['SVO'], label='SVO', color='purple')
                    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
                    ax3.legend()
                    ax3.grid(True)
            except:
                raise ValueError("Debe haber 3 ejes para plotear")
        else:
            try:
                if put_plot:
                    ax1, ax2, ax3 = axes
            except:
                raise ValueError("Debe haber 2 ejes para plotear")
        
        # Ploteamos medias y precio cierre.
        if put_plot:
            ax1.plot(self.df.index, self.df['Close'], label='Close', color='black')
            ax2.plot(self.df.index, self.df[f'RSI'], label=f'RSI({period_RSI})', color='purple')
            ax2.axhline(rsi_zones[0], color='orange', linewidth=0.5, linestyle='--')
            ax2.axhline(rsi_zones[1], color='orange', linewidth=0.5, linestyle='--')
            ax2.legend()
            ax2.grid(True)

        
        # Inicializamos variables
        capital = capital_inicial
        self.show_print = show_print
        self.comision = comision
        self.reiniciar_variables()
        
        # Determinar las señales de compra y venta basadas en las BB
        for i in range(1, len(self.df)):
            # Estrategia de long: 
            if self.Señal_rsi(i, rsi_zones=rsi_zones)=="sobreventa"  and self.Señal_MM(i,col_MM) in ("cruce_alcista","alcista") and self.tipo_entrada==None:
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.abrir_posicion("long",capital,i)
                elif put_SVO==False :
                    capital = self.abrir_posicion("long",capital,i)
                continue
            # Estrategia de short: 
            elif self.Señal_rsi(i, rsi_zones=rsi_zones)=="sobrecompra" and self.Señal_MM(i,col_MM) in ("cruce_bajista","bajista") and self.tipo_entrada==None:
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.abrir_posicion("short",capital,i)
                elif put_SVO==False :
                    capital = self.abrir_posicion("short",capital,i)
                continue
            # Estrategia de Salida long:
            elif self.Señal_rsi(i, rsi_zones=rsi_zones)=="sobreventa" and self.tipo_entrada=="long":
                if put_SVO and self.df["SVO"][i] <0:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
                elif put_SVO==False:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
            # Estrategia de Salida short:
            elif self.Señal_rsi(i, rsi_zones=rsi_zones)=="sobrecompra" and self.tipo_entrada=="short":
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
                elif put_SVO==False:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
            # Calcular el pseudo - rendimiento
            self.calcular_pseudorendimiento(capital,i)
            
        # Ploteamos las señales
        if put_plot:
            self.Plot_senales(ax1,period_RSI)
            ax1.set_title(f'{symbol} con SVO')
            ax1.set_xlabel('Fecha')
            ax1.set_ylabel('Precio')
            ax1.legend()
            ax1.grid(True)
        if show_print:
            print(f"Capital final: {capital}")
        
        if put_SVO and put_plot:
            self.Plot_Cartera(ax=ax4)
        elif put_plot:
            self.Plot_Cartera(ax=ax3)
        return capital, period_RSI, period_SVO, self.operaciones, rsi_zones
    
    
    def Plot_Estrategia_Stocastico(self, df, symbol, periodos_stochastic=[14,3], period_SVO=20, periodos_mm=[7,30],
                            capital_inicial=100, comision=0.001, show_print=False, put_SVO=False,  axes=None,
                            rsi_zones=[20,80], put_plot=False, col_MM="EMA"):
        # Generamos el indicador
        # Generamos el RSi
        self.df = df.copy()
        self.Put_indicators({"MM":periodos_mm,"SVO":period_SVO, "estocastico":periodos_stochastic})
        
        if put_SVO:
            try:
                if put_plot:
                    ax1, ax2, ax3, ax4 = axes
                    ax3.plot(self.df.index, self.df['SVO'], label='SVO', color='purple')
                    ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
                    ax3.legend()
                    ax3.grid(True)
            except:
                raise ValueError("Debe haber 3 ejes para plotear")
        else:
            try:
                if put_plot:
                    ax1, ax2, ax3 = axes
            except:
                raise ValueError("Debe haber 2 ejes para plotear")
        
        # Ploteamos medias y precio cierre.
        if put_plot:
            ax1.plot(self.df.index, self.df['Close'], label='Close', color='black')
            ax2.plot(self.df.index, self.df[f'%K'], label=f'%k({periodos_stochastic[0]})', color='blue')
            ax2.plot(self.df.index, self.df[f'%D'], label=f'%D({periodos_stochastic[0]})', color='orange')
            ax2.axhline(rsi_zones[0], color='orange', linewidth=0.5, linestyle='--')
            ax2.axhline(rsi_zones[1], color='orange', linewidth=0.5, linestyle='--')
            ax2.legend()
            ax2.grid(True)

        
        # Inicializamos variables
        capital = capital_inicial
        self.show_print = show_print
        self.comision = comision
        self.reiniciar_variables()
        
        # Determinar las señales de compra y venta basadas en las 
        for i in range(1, len(self.df)):
            # Estrategia de long:
            if self.Señal_rsi(i, rsi_zones=rsi_zones, col_RSI="%D")=="sobreventa"  and self.Señal_MM(i,col_MM) in ("cruce_alcista","alcista") and self.tipo_entrada==None:
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.abrir_posicion("long",capital,i)
                elif put_SVO==False :
                    capital = self.abrir_posicion("long",capital,i)
                continue
            # Estrategia de short: 
            elif self.Señal_rsi(i, rsi_zones=rsi_zones,col_RSI="%D")=="sobrecompra" and self.Señal_MM(i,col_MM) in ("cruce_bajista","bajista") and self.tipo_entrada==None:
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.abrir_posicion("short",capital,i)
                elif put_SVO==False :
                    capital = self.abrir_posicion("short",capital,i)
                continue
            # Estrategia de Salida long:
            elif self.Señal_rsi(i, rsi_zones=rsi_zones, col_RSI="%D")=="sobreventa" and self.tipo_entrada=="long":
                if put_SVO and self.df["SVO"][i] <0:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
                elif put_SVO==False:
                    capital = self.cerrar_posicion("long", capital, i, psuedo_cierre=False)
                    continue
            # Estrategia de Salida short:
            elif self.Señal_rsi(i, rsi_zones=rsi_zones, col_RSI="%D")=="sobrecompra" and self.tipo_entrada=="short":
                if put_SVO and self.df["SVO"][i] >0:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
                elif put_SVO==False:
                    capital = self.cerrar_posicion("short", capital, i, psuedo_cierre=False)
                    continue
            # Calcular el pseudo - rendimiento
            self.calcular_pseudorendimiento(capital,i)
            
        # Ploteamos las señales
        if put_plot:
            self.Plot_senales(ax1,periodos_stochastic)
            ax1.set_title(f'{symbol} con SVO')
            ax1.set_xlabel('Fecha')
            ax1.set_ylabel('Precio')
            ax1.legend()
            ax1.grid(True)
        if show_print:
            print(f"Capital final: {capital}")
        
        if put_SVO and put_plot:
            self.Plot_Cartera(ax=ax4)
        elif put_plot:
            self.Plot_Cartera(ax=ax3)
        return capital, periodos_stochastic, period_SVO, self.operaciones
    
    def probar_fibonacci(self,df, symbol, col_use="Close", axes=None, col_MM="EMA",
                         periodos_fibonacci=15,period_SVO=20, periodos_mm=[8,55],
                         capital_inicial=100, comision=0.001, show_print=False,
                         put_Plot=False):
        self.df = df.copy()
        self.Put_indicators({"MM":periodos_mm,"SVO":period_SVO})
        
        if put_Plot:
            ax1, ax2 = axes
            axes[0].plot(self.df.index, self.df['Close'], label='Close', color='black')
            axes[0].plot(self.df.index, self.df[f'{col_MM}0'], label=f'{col_MM}{periodos_mm[0]}', color='blue')
            axes[0].plot(self.df.index, self.df[f'{col_MM}1'], label=f'{col_MM}{periodos_mm[1]}', color='orange')

        # Inicializamos variables
        capital = capital_inicial
        nivel_fib_entrada = 0.786
        precio_entrada = None
        self.show_print = show_print
        self.comision = comision
        self.reiniciar_variables()
        
        # Para poner adecuadamente los fibonacci
        x_total = len(df)
        niveles_fibonacci = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        
        # Determinar las señales de compra y venta basadas en las 
        for i in range(1, len(self.df)):
            señal = self.Señal_MM(i,col_MM="EMA")
            precio_actual = self.df["Close"][i]
            
            if señal == "cruce_alcista":
                
                # Ponemos fibonacci alcista
                index_ini = 0 if i - periodos_fibonacci <0 else i - periodos_fibonacci
                precios_fib = self.df[col_use].iloc[index_ini:i]
                print(index_ini,i, precios_fib.shape)
                precios_fibonacci = calcular_fibonacci(precios_fib, reversa=False, niveles_fibonacci=niveles_fibonacci)
                fechas_fib = precios_fib.index
                
                # Buscamos el precio de entrada 
                precio_entrada = precios_fibonacci[f'Nivel {round((nivel_fib_entrada)*100,2)}%']
                # Ponemos una orden 
                self.orden="long"
                # Cerramos el short en caso de existir
                if self.tipo_entrada =="short":
                    capital = self.cerrar_posicion("short",capital,i,False)
                    
                # Ploteamos el fibonacci
                               
                       
                # Graficar los niveles de Fibonacci desde la fecha donde comienzan los últimos periodos_fib
                if put_Plot:
                    #graficar_fibonacci(precios_fib,ax1, periodos_fibonacci, reversa=False)
                    for nivel, precio in precios_fibonacci.items():
                        ax1.axhline(y=precio, xmin=(i-periodos_fibonacci)/x_total, xmax=(i)/x_total, linestyle='--', alpha=0.6, color='purple')
                        # Añadir la etiqueta en la línea, en el eje x desde la primera fecha de fibonacci
                        ax1.text(fechas_fib[0], precio, f"{nivel} ({precio})", color='black', fontsize=10, verticalalignment='center')
                
                continue
                
                #self.poner_orden(i,tipo="long",entrada=precio_entrada,tp=None, sl=None)
            elif señal == "cruce_bajista":
                # Ponemos fibonacci bajista
                index_ini = 0 if i < periodos_fibonacci else i - periodos_fibonacci
                precios_fib = self.df[col_use].iloc[:i]
                print(index_ini,i, precios_fib.shape)
                precios_fibonacci = calcular_fibonacci(precios_fib, reversa=True, niveles_fibonacci=niveles_fibonacci)
                fechas_fib = precios_fib.index
                
                # Buscamos el precio de entrada 
                precio_entrada = precios_fibonacci[f'Nivel {round((nivel_fib_entrada)*100,2)}%']
                # Ponemos una orden 
                self.orden="short"
                # Cerramos el long en caso de existir
                if self.tipo_entrada =="long":
                    capital = self.cerrar_posicion("long",capital,i,False)
                    self.precio_entrada=None
                # Graficar los niveles de Fibonacci desde la fecha donde comienzan los últimos periodos_fib
                if put_Plot:
                    #graficar_fibonacci(precios_fib,ax1, periodos_fibonacci, reversa=True)
                    for nivel, precio in precios_fibonacci.items():
                        ax1.axhline(y=precio, xmin=(i-periodos_fibonacci)/x_total, xmax=(i)/x_total, linestyle='--', alpha=0.6, color='purple')
                        # Añadir la etiqueta en la línea, en el eje x desde la primera fecha de fibonacci
                        ax1.text(fechas_fib[0], precio, f"{nivel} ({precio})", color='black', fontsize=10, verticalalignment='center')
                    
                continue
                
            if self.orden=="long" and precio_entrada!=None:
                if precio_actual<= precio_entrada:
                    capital = self.abrir_posicion("long",capital,i)
            elif self.orden=="short" and precio_entrada!=None:
                if precio_actual>= precio_entrada:
                    capital = self.abrir_posicion("short",capital,i)
            # Pseudo-Cerramos el long en caso de existir
            if self.tipo_entrada =="long":
                self.cerrar_posicion("long",capital,i,True)
                continue
            # Pseudo-Cerramos el short en caso de existir
            elif self.tipo_entrada =="short":
                self.cerrar_posicion("short",capital,i,True)
                continue
            self.rows.append((capital,self.df.index[i],"",self.df.Close[i],0))
                
        # Ploteamos las señales
        if put_Plot:
            self.Plot_senales(axes[0],periodos_mm)
            axes[0].set_title(f'{symbol} con Fibonacci')
            axes[0].set_xlabel('Fecha')
            axes[0].set_ylabel('Precio')
            axes[0].legend()
            axes[0].grid(True)
            self.Plot_Cartera(ax=axes[1])
            
        return capital