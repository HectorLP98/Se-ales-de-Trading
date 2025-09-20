from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Utilidades de ventana deslizante (O(1))
# -----------------------------
class _RollingExtreme:
    """
    Mantiene el mínimo o máximo de una ventana deslizante en O(1) por paso.
    Si mode == 'min' mantiene mínimos; si mode == 'max' mantiene máximos.
    Guarda índices para poder expulsar elementos que salen de la ventana.
    """
    def __init__(self, arr: np.ndarray, window: int, mode: str):
        assert mode in ("min", "max")
        self.arr = arr
        self.window = int(window)
        self.mode = mode
        self.q = deque()  # guarda índices

    def _cmp(self, a, b):
        return a <= b if self.mode == "min" else a >= b

    def push(self, i: int):
        # expulsar por la cola mientras el nuevo valor "mejora" el extremo
        val = self.arr[i]
        q = self.q
        arr = self.arr
        while q and self._cmp(val, arr[q[-1]]):
            q.pop()
        q.append(i)
        # expulsar por la cabeza si sale de la ventana
        wstart = i - self.window + 1
        while q and q[0] < wstart:
            q.popleft()

    def peek(self) -> float:
        if not self.q:
            return np.nan
        return float(self.arr[self.q[0]])


@dataclass
class FVG:
    fecha_inicio: object
    fecha_fin: object
    rango: Tuple[float, float]
    tocada: bool = False
    fecha_tocada: Optional[object] = None


class AnalizadorVelas:
    """
    Analizador optimizado para OHLC con una sola pasada del dataset.

    Genera y adjunta al DataFrame los campos:
      - FVGs (0/1)
      - Ruptura_Alcista (0/1)
      - Ruptura_Bajista (0/1)
      - Tipo_Ruptura ("", "Mecha", "Vela")

    Además produce resúmenes en el método analizar().
    """

    # codificación interna para ahorrar memoria (se mapea a string al final)
    _TIPO_NONE = np.int8(0)
    _TIPO_MECHA = np.int8(1)
    _TIPO_VELA = np.int8(2)
    _TIPO_MAP = {0: "", 1: "Mecha", 2: "Vela"}
    _TIPO_VELA_MAP = {
            0: "Plana",
            1: "Martillo",
            2: "Martillo_Inv",
            3: "Hang_Man",
            4: "Estrella_Fug",
            5: "Doji",
            6: "Cuerpo_Lleno",
            7: ""
        }

    def __init__(self, df: pd.DataFrame, rango_analisis: Tuple[int, int] = (10, 100)):
        # Guardar referencia; no copiamos para ahorrar memoria
        self.df_original = df
        self.rango_analisis = tuple(int(x) for x in rango_analisis)

        # Arrays NumPy (sin copia cuando es posible)
        self.arr_open_full = df["Open"].to_numpy(copy=False)
        self.arr_high_full = df["High"].to_numpy(copy=False)
        self.arr_low_full = df["Low"].to_numpy(copy=False)
        self.arr_close_full = df["Close"].to_numpy(copy=False)
        self.arr_index_full = df.index.to_numpy()

        self.n = len(self.arr_open_full)
        if not (self.n == len(self.arr_high_full) == len(self.arr_low_full) == len(self.arr_close_full)):
            raise ValueError("Las columnas OHLC deben tener la misma longitud")

        # Resultados detallados
        self.fvg_dict: Dict[int, Dict] = {}
        self.fvg_abiertas: List[int] = []  # ids de FVG no tocadas

        # Conteos
        self.conteo: Dict[str, int] = {
            "Iteraciones": 0,
            "Mechas alcistas": 0,
            "Mechas bajistas": 0,
            "Velas alcistas": 0,
            "Velas bajistas": 0,
        }

        # Prealocación de columnas de salida (más eficiente que listas de Python)
        self._col_fvg = np.zeros(self.n, dtype=np.int8)
        self._col_rup_up = np.zeros(self.n, dtype=np.int8)
        self._col_rup_dn = np.zeros(self.n, dtype=np.int8)
        self._col_tipo = np.zeros(self.n, dtype=np.int8)  # codificada

    # -----------------------------
    # Métodos internos
    # -----------------------------
    def _clasificar_vela(self, i: int) -> str:
        return "green" if self.arr_close_full[i] >= self.arr_open_full[i] else "red"

    def _detectar_fvg(self, i: int, secuencial: int) -> int:
        """Detecta FVG en i vs i-2. Devuelve 1 si hay FVG (para marcar columna) y registra en fvg_dict."""
        if i < 2:
            return 0
        low_i = self.arr_low_full[i]
        high_i = self.arr_high_full[i]
        high_i_2 = self.arr_high_full[i - 2]
        low_i_2 = self.arr_low_full[i - 2]

        if low_i > high_i_2:  # FVG alcista
            self.fvg_dict[secuencial] = FVG(
                fecha_inicio=self.arr_index_full[i - 2],
                fecha_fin=self.arr_index_full[i],
                rango=(float(high_i_2), float(low_i)),
            ).__dict__
            self.fvg_abiertas.append(secuencial)
            return 1
        elif high_i < low_i_2:  # FVG bajista
            self.fvg_dict[secuencial] = FVG(
                fecha_inicio=self.arr_index_full[i - 2],
                fecha_fin=self.arr_index_full[i],
                rango=(float(high_i), float(low_i_2)),
            ).__dict__
            self.fvg_abiertas.append(secuencial)
            return 1
        return 0

    def _overlap(self, low: float, high: float, a: float, b: float) -> bool:
        lo, hi = (a, b) if a <= b else (b, a)
        return max(low, lo) <= min(high, hi)

    def _verificar_testeo_fvg(self, i: int):
        """Marca FVGs abiertas como 'tocada' si el rango se solapa con [low_i, high_i].
        Se itera sobre una lista probablemente pequeña.
        """
        if not self.fvg_abiertas:
            return
        low_i = self.arr_low_full[i]
        high_i = self.arr_high_full[i]
        fecha = self.arr_index_full[i]
        # Iterar por copia de la lista porque podemos modificar la original
        for fvg_id in self.fvg_abiertas[:]:
            fvg = self.fvg_dict[fvg_id]
            a, b = fvg["rango"]
            if self._overlap(low_i, high_i, a, b):
                fvg["tocada"] = True
                fvg["fecha_tocada"] = fecha
                # sacar de abiertas
                self.fvg_abiertas.remove(fvg_id)

    def _contar_rupturas(self, i: int, min_low: float, max_high: float):
        """
        Aplica las reglas del usuario para clasificar ruptura y actualizar conteos.
        Rellena columnas Ruptura_Alcista, Ruptura_Bajista y Tipo_Ruptura (codificada).
        """
        open_ = self.arr_open_full[i]
        high = self.arr_high_full[i]
        low = self.arr_low_full[i]
        close = self.arr_close_full[i]

        tipo_vela = self._clasificar_vela(i)

        # Inicializar por si no hay ruptura
        rup_up = 0
        rup_dn = 0
        tipo_cod = self._TIPO_NONE

        # Orden de evaluación y mapeo a campos
        if (high > max_high) and ((tipo_vela == "green" and close <= max_high) or (tipo_vela == "red" and open_ <= max_high)):
            # Mecha alcista
            self.conteo["Mechas alcistas"] += 1
            rup_up = 1
            tipo_cod = self._TIPO_MECHA
        elif (low < min_low) and ((tipo_vela == "red" and close >= min_low) or (tipo_vela == "green" and open_ >= min_low)):
            # Mecha bajista
            self.conteo["Mechas bajistas"] += 1
            rup_dn = 1
            tipo_cod = self._TIPO_MECHA
        elif (close > max_high) or (open_ > max_high) or (high > max_high):
            # Vela alcista
            self.conteo["Velas alcistas"] += 1
            rup_up = 1
            tipo_cod = self._TIPO_VELA
        elif (close < min_low) or (open_ < min_low):
            # Vela bajista
            self.conteo["Velas bajistas"] += 1
            rup_dn = 1
            tipo_cod = self._TIPO_VELA

        self._col_rup_up[i] = rup_up
        self._col_rup_dn[i] = rup_dn
        self._col_tipo[i] = tipo_cod

        # ---------------- Clasificación de velas ----------------
    
    def _color_vela(self, open_, close):
        """Devuelve 1 si es alcista, -1 si es bajista, 0 si es neutra."""
        if close > open_:
            return 1
        elif close < open_:
            return -1
        else:
            return 0

    def _tipo_vela(self, open_, high, low, close, pt1=0.5, pt2=0.05, pt3=0.4):
        """
        Clasifica la vela según la relación de cuerpo y mechas.
        Retorna: str -> "Martillo", "Martillo_Inv", "Hang_Man", "Estrella_Fug",
                         "Doji", "Cuerpo_Lleno", "Plana", "N/A"
        """
        pt = high - low
        pmi = min(close, open_) - low
        pms = high - max(close, open_)
        pcv = abs(close - open_)
        color = self._color_vela(open_, close)
        
        if pt == 0:
            return 0
        elif pmi >= pt1 * pt and pms <= pt2 * pt and color == 1:
            return 1
        elif pmi <= pt2 * pt and pms >= pt1 * pt and color == 1:
            return 2
        elif pmi >= pt1 * pt and pms <= pt2 * pt and color == -1:
            return 3
        elif pmi <= pt2 * pt and pms >= pt1 * pt and color == -1:
            return 4
        elif pmi >= pt3 * pt and pms >= pt3 * pt:
            return 5
        elif pcv >= 0.95 * pt:
            return 6
        else:
            return 7

    # -----------------------------
    # Resúmenes vectorizados sobre columnas nuevas
    # -----------------------------
    def _resumen_sobre_campos(self) -> Dict[str, Dict]:
        # Trabajar con vistas NumPy para velocidad
        fvg = self._col_fvg
        rup_up = self._col_rup_up
        rup_dn = self._col_rup_dn
        tipo = self._col_tipo

        total = int(self.conteo["Iteraciones"]) if self.conteo["Iteraciones"] else int(max(len(fvg) - 2, 0))
        total_fvg = int(fvg.sum())
        total_no_tocadas = int(sum(1 for k in self.fvg_dict if not self.fvg_dict[k]["tocada"]))
        total_tocadas = total_fvg - total_no_tocadas
        pct_tocadas = float(100.0 * total_tocadas / total_fvg) if total_fvg else 0.0

        dict_fvg_no_tocadas = {
            "total_fvg": total_fvg,
            "total_no_tocadas": total_no_tocadas,
            "pct_tocadas": round(pct_tocadas, 2),
            "ultimas_fvg_info": list(self.fvg_dict.values())[-5:],
        }

        # Resumen rupturas
        resumen_rupturas = {
            "Total_Rupturas_Alcistas": int(rup_up.sum()),
            "Total_Rupturas_Bajistas": int(rup_dn.sum()),
            "Total_Mechas": int(np.sum(tipo == self._TIPO_MECHA)),
            "Total_Velas": int(np.sum(tipo == self._TIPO_VELA)),
        }

        # Std/porcentajes basados en campos de ruptura (definición operativa y barata)
        precio_actual = float(self.arr_close_full[-1]) if self.n else np.nan
        porc_alc = float(100.0 * resumen_rupturas["Total_Rupturas_Alcistas"] / total) if total else 0.0
        porc_baj = float(100.0 * resumen_rupturas["Total_Rupturas_Bajistas"] / total) if total else 0.0
        dict_std_min_max = {
            "Precio_Actual": precio_actual,
            "Porcentaje_Alcista": round(porc_alc, 2),
            "Porcentaje_Bajista": round(porc_baj, 2),
            "Diferencia_Porcentajes": round(abs(porc_alc - porc_baj), 2),
        }

        return dict_fvg_no_tocadas, resumen_rupturas, dict_std_min_max

    # -----------------------------
    # Método principal
    # -----------------------------
    def analizar(self) -> Dict[str, object]:
        if self.n < 3:
            # Dataset demasiado corto, devolvemos estructuras vacías pero consistentes
            self.df_original = self.df_original.assign(
                FVGs=self._col_fvg,
                Ruptura_Alcista=self._col_rup_up,
                Tipo_Ruptura=np.array([self._TIPO_MAP[int(x)] for x in self._col_tipo], dtype=object),
                Ruptura_Bajista=self._col_rup_dn,
            )
            dict_fvg_no_tocadas, resumen_rupturas, dict_std_min_max = self._resumen_sobre_campos()
            return {
                "FVGs": self.fvg_dict,
                "FVGs_no_tocadas": dict_fvg_no_tocadas,
                "Conteo_Rupturas": self.conteo,
                "Rango_Min_Max": (np.nan, np.nan),
                "Std_Min_Max": dict_std_min_max,
                "Resumen_Swings": {},  # el usuario puede implementar su lógica propia
                "Velas": {},
            }

        # Ventana para min/max (usamos el segundo elemento del rango como referencia)
        win = max(self.rango_analisis[1], 3)
        roll_min = _RollingExtreme(self.arr_low_full, window=win, mode="min")
        roll_max = _RollingExtreme(self.arr_high_full, window=win, mode="max")

        # Inicializar estructuras previas a i=2
        secuencial = 0
        global_min = float("inf")
        global_max = float("-inf")
        
        # Tipo de vela (string) para cada índice
        tipo_vela_arr = np.zeros(self.n, dtype=np.int8)  # codificada

        # Pre-cargar ventana hasta i-1 para que min/max excluyan la vela i
        # Empezamos el procesamiento en i=2 por la regla de FVG
        for j in range(0, 2):
            roll_min.push(j)
            roll_max.push(j)
            global_min = min(global_min, float(self.arr_low_full[j]))
            global_max = max(global_max, float(self.arr_high_full[j]))

        # Bucle principal: O(n)
        for i in range(2, self.n):
            self.conteo["Iteraciones"] += 1
            tipo_vela_arr[i] = self._tipo_vela(self.arr_open_full[i], self.arr_high_full[i], self.arr_low_full[i], self.arr_close_full[i])

            # min/max del histórico reciente (excluye vela i)
            min_low = roll_min.peek()
            max_high = roll_max.peek()

            # 1) Detectar FVG en i
            fvg_flag = self._detectar_fvg(i, secuencial)
            if fvg_flag:
                self._col_fvg[i] = 1
                secuencial += 1

            # 2) Verificar si alguna FVG abierta es testeada por la vela i
            self._verificar_testeo_fvg(i)

            # 3) Contar rupturas y escribir columnas
            self._contar_rupturas(i, min_low=min_low, max_high=max_high)

            # 4) Clasificación individual de vela (si el usuario quiere guardarla aparte)
            #   Aquí evitamos crear dicts por vela para ahorrar memoria. Si se desea, se puede
            #   construir externamente con el índice que interese.

            # 5) Actualizar extremos globales y ventana (incluir i en rolling para i+1)
            global_min = min(global_min, float(self.arr_low_full[i]))
            global_max = max(global_max, float(self.arr_high_full[i]))
            roll_min.push(i)
            roll_max.push(i)

        # Mapear tipo codificado a string solo una vez (barato)
        tipo_str = np.empty(self.n, dtype=object)
        tipo_str_vela = np.empty(self.n, dtype=object)
        for k in range(self.n):
            tipo_str[k] = self._TIPO_MAP[int(self._col_tipo[k])]
            tipo_str_vela[k] = self._TIPO_VELA_MAP[int(tipo_vela_arr[k])]
        # Adjuntar columnas al DataFrame original (no copiamos datos OHLC)
        self.df_original["FVGs"] = self._col_fvg
        self.df_original["Ruptura_Alcista"] = self._col_rup_up
        self.df_original["Tipo_Ruptura"] = tipo_str
        self.df_original["Ruptura_Bajista"] = self._col_rup_dn
        self.df_original["Tamanio_Vela"] = self.arr_high_full - self.arr_low_full
        self.df_original["Tipo_Vela"] = tipo_str_vela

        # Resúmenes
        dict_fvg_no_tocadas, resumen_rupturas, dict_std_min_max = self._resumen_sobre_campos()

        # (Opcional) resumen de swings: dejar como hook para que el usuario lo personalice
        resumen_swings = {
            "Total_Swing_Alcistas": 0,
            "Total_Swing_Bajistas": 0,
            "Fecha_Ultimo_Swing_Alcista": None,
            "Precio_Ultimo_Swing_Alcista": None,
            "Fecha_Ultimo_Swing_Bajista": None,
            "Precio_Ultimo_Swing_Bajista": None,
        }

        return {
            "FVGs": self.fvg_dict,
            "FVGs_no_tocadas": dict_fvg_no_tocadas,
            "Conteo_Rupturas": self.conteo,
            "Rango_Min_Max": (global_min, global_max),
            "Std_Min_Max": dict_std_min_max,
            "Resumen_Swings": resumen_swings,
            "Velas": {},  # evitar objetos por vela para ahorrar memoria (se puede reconstruir si se requiere)
        }


# Uso rápido (ejemplo):
# analyzer = AnalizadorVelas(df, rango_analisis=(10, 100))
# resumen = analyzer.analizar()
# df_result = analyzer.df_original  # ya incluye columnas FVGs, Ruptura_Alcista, Tipo_Ruptura, Ruptura_Bajista
