from datetime import datetime 

def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds
    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str
    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms

def fecha_a_milisegundos(fecha_str, fecha_format:str='%Y-%m-%d %H:%M:%S'):
    '''
    (Function)
        Esta funcion ingresa una fecha str y su formato para retornar 
        la fecha en milisegundos
    (Parameters)
        - fecha_str: Fecha a transformar en str
        - fecha_format: Formato de fecha, ejemlo para 28-08-2023 "%d-%m-%Y"
    (Returns)
        int: fecha en milisegundos
    (Author)
        Hector Limon
    '''
    if isinstance(fecha_str,str):
        # Fecha en formato 'año-mes-día'
        fecha_obj = datetime.strptime(fecha_str, fecha_format)

    elif isinstance(fecha_str,datetime):
        fecha_obj = fecha_str
    
    else:
        raise ValueError("Error en fecha_Str, asegurese que sea str o datetime")
    # Calcular la diferencia desde la época (epoch) en segundos y luego a milisegundos
    milisegundos = int(fecha_obj.timestamp() * 1000)
    return milisegundos

def contar_dias_desde_año(fecha_str='2018-01-01',fecha_format:str='%Y-%m-%d'):
    '''
    (Function)
        Esta funcion cuenta los dias que hay desde una fecha especifica al dia actual
        se toma en cuenta la fecha en que se ejecta
    (Parameters)
        - fecha_str: Fecha inicio
        - fecha_format: Formato de fecha, ejemlo para 28-08-2023 "%d-%m-%Y"
    (Returns)
        int: Numero de dias transcurridos
    (Author)
        Hector Limon
    '''
    if isinstance(fecha_str,str):
        # Fecha en formato 'año-mes-día'
        fecha_especifica = datetime.strptime(fecha_str, fecha_format)

    elif isinstance(fecha_str,datetime):
        fecha_especifica = fecha_str
    
    else:
        raise ValueError("Error en fecha_Str, asegurese que sea str o datetime")
    
    fecha_actual = datetime.now()
    diferencia = fecha_actual - fecha_especifica
    dias_transcurridos = diferencia.days
    return dias_transcurridos


def get_credentials(file:str="Data/vacio.txt"):
    texto = open(file,'r')
    lines = texto.readlines()
    apy_key = lines[0].split('=')[-1].strip()
    secret_key = lines[1].split('=')[-1].strip()
    my_sql_pass = lines[2].split('=')[-1].strip()
    #my_sql_pass = my_sql_pass[my_sql_pass.find('=')+1:]
    return apy_key, secret_key, my_sql_pass