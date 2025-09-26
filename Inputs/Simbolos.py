def Obtener_Simbolos_Binance(client,mercado='spot',filtro=None):
    """
    Obtiene los símbolos disponibles en Binance para el mercado spot.
    
    Args:
        client: Cliente de Binance inicializado con las credenciales API.
        
    Returns:
        list: Lista de símbolos disponibles en el mercado spot de Binance.
    """
    if mercado.lower() == 'futuros':
        # Obtener símbolos de futuros
        symbols_Binance = [symbol['symbol'] for symbol in client.exchange_info()['symbols'] if symbol['contractType'] == 'PERPETUAL']
        r
    elif mercado.lower() == 'spot':
        # Obtener símbolos de spot
        symbols_Binance = [symbol['symbol'] for symbol in client.exchange_info()['symbols'] if symbol['status'] == 'TRADING' and symbol['isSpotTradingAllowed']]
    
    simbolos = symbols_Binance.copy()
    if filtro:
        simbolos = [s for s in simbolos if filtro.lower() in s.lower()]
    
    return simbolos