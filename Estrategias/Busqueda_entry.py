import numpy as np

def greedy_trades(prices):
    signals = [None] * len(prices)
    in_trade = False
    
    for i in range(1, len(prices)-1):
        if not in_trade:
            # Buscar mínimo local (entrada)
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                signals[i] = 1  # entrada
                in_trade = True
        else:
            # Buscar máximo local (salida)
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                signals[i] = -1  # salida
                in_trade = False
            else:
                signals[i] = 0  # mantener
    
    return signals


def greedy_trades_tp_sl(prices, tp=0.05, sl=0.02, t_max=10):
    """
    Busca trades en una serie de precios con reglas greedy:
    - Entrada en mínimos locales.
    - Salida cuando se alcanza TP, SL o tiempo máximo.
    
    Params:
        prices : list o np.array de precios
        tp     : take profit (ej. 0.05 = 5%)
        sl     : stop loss (ej. 0.02 = 2%)
        t_max  : duración máxima del trade (en pasos)
    """
    signals = [None] * len(prices)
    in_trade = False
    entry_price = None
    entry_index = None

    for i in range(1, len(prices)-1):
        if not in_trade:
            # Detectar mínimo local como punto de entrada
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                signals[i] = 1   # entrada
                in_trade = True
                entry_price = prices[i]
                entry_index = i
        else:
            # Calcular rendimiento actual
            rendimiento = (prices[i] - entry_price) / entry_price
            duracion = i - entry_index

            # Revisar condiciones de salida
            if rendimiento >= tp:
                signals[i] = -1  # take profit
                in_trade = False
            elif rendimiento <= -sl:
                signals[i] = -1  # stop loss
                in_trade = False
            elif duracion >= t_max:
                signals[i] = -1  # salida por tiempo
                in_trade = False
            else:
                signals[i] = 0  # mantener

    return signals

def greedy_trades_trailing(prices, tp=0.10, sl=0.03, trailing=0.05):
    """
    Estrategia greedy mejorada:
    - Entrada en mínimos locales
    - Salida si toca SL, TP o trailing stop dinámico
    
    Params:
        prices   : lista de precios
        tp       : take profit objetivo inicial (ej 0.10 = 10%)
        sl       : stop loss inicial (ej 0.03 = 3%)
        trailing : margen de trailing stop (ej 0.05 = 5%)
    """
    signals = [None] * len(prices)
    in_trade = False
    entry_price = None
    peak_price = None

    for i in range(1, len(prices)-1):
        if not in_trade:
            # Buscar entrada en mínimo local
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                signals[i] = 1
                in_trade = True
                entry_price = prices[i]
                peak_price = prices[i]
        else:
            # Actualizar máximo alcanzado desde la entrada
            peak_price = max(peak_price, prices[i])
            rendimiento = (prices[i] - entry_price) / entry_price
            drawdown = (prices[i] - peak_price) / peak_price

            # Condiciones de salida
            if rendimiento >= tp:
                signals[i] = -1   # alcanzó TP
                in_trade = False
            elif rendimiento <= -sl:
                signals[i] = -1   # stop loss
                in_trade = False
            elif drawdown <= -trailing:
                signals[i] = -1   # retroceso tras máximo (trailing stop)
                in_trade = False
            else:
                signals[i] = 0    # mantener

    return signals

