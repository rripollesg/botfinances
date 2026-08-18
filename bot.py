import html
import os
import sys
import time

import pandas as pd
import requests
import yfinance as yf
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "TU_TOKEN_DE_TELEGRAM_LOCAL")
CHAT_ID = os.getenv("CHAT_ID", "TU_CHAT_ID_LOCAL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# auto | yfinance | metatrader
# - yfinance: modo standalone (sin plataforma externa)
# - metatrader: datos desde MT5 local
# - auto: intenta MT5 y, si falla, usa yfinance
FUENTE_DATOS = os.getenv("FUENTE_DATOS", "auto").lower()
MT5_PATH = os.getenv("MT5_PATH", "")
MT5_SUFFIX = os.getenv("MT5_SUFFIX", "")

RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 70
BACKTEST_INITIAL_CAPITAL = float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000"))
BACKTEST_COMMISSION_PCT = float(os.getenv("BACKTEST_COMMISSION_PCT", "0.001"))
BACKTEST_SPREAD_PCT = float(os.getenv("BACKTEST_SPREAD_PCT", "0.0005"))
BACKTEST_STOP_LOSS_PCT = float(os.getenv("BACKTEST_STOP_LOSS_PCT", "0.08"))
BACKTEST_TAKE_PROFIT_PCT = float(os.getenv("BACKTEST_TAKE_PROFIT_PCT", "0.15"))
BACKTEST_OUTPUT_DIR = os.getenv("BACKTEST_OUTPUT_DIR", "backtests")

PORTAFOLIO = [
    # Tech / IA (núcleo)
    {"ticker": "NVDA", "nombre": "Nvidia", "tipo": "Acción"},
    {"ticker": "AAPL", "nombre": "Apple", "tipo": "Acción"},
    {"ticker": "AMD", "nombre": "AMD", "tipo": "Acción"},
    {"ticker": "AVGO", "nombre": "Broadcom", "tipo": "Acción"},
    {"ticker": "TSM", "nombre": "TSMC", "tipo": "Acción"},
    # Índices
    {"ticker": "SPY", "nombre": "S&P 500", "tipo": "ETF"},
    {"ticker": "QQQ", "nombre": "Nasdaq 100", "tipo": "ETF"},
    {"ticker": "IWM", "nombre": "Russell 2000", "tipo": "ETF"},
    # Rotación sectorial
    {"ticker": "XLV", "nombre": "Salud", "tipo": "ETF"},
    {"ticker": "XLF", "nombre": "Financiero", "tipo": "ETF"},
    {"ticker": "XLE", "nombre": "Energía", "tipo": "ETF"},
    {"ticker": "LLY", "nombre": "Eli Lilly", "tipo": "Acción"},
    # Cripto (reducido)
    {"ticker": "BTC-USD", "nombre": "Bitcoin", "tipo": "Cripto"},
]

TRADINGVIEW_SYMBOLS = {
    "NVDA": "NASDAQ:NVDA",
    "AAPL": "NASDAQ:AAPL",
    "AMD": "NASDAQ:AMD",
    "AVGO": "NASDAQ:AVGO",
    "TSM": "NYSE:TSM",
    "SPY": "AMEX:SPY",
    "QQQ": "NASDAQ:QQQ",
    "IWM": "AMEX:IWM",
    "XLV": "AMEX:XLV",
    "XLF": "AMEX:XLF",
    "XLE": "AMEX:XLE",
    "LLY": "NYSE:LLY",
    "BTC-USD": "BINANCE:BTCUSDT",
}

# Símbolos habituales en MetaTrader 5 (pueden variar según broker)
METATRADER_SYMBOLS = {
    "NVDA": "NVDA",
    "AAPL": "AAPL",
    "AMD": "AMD",
    "AVGO": "AVGO",
    "TSM": "TSM",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "XLV": "XLV",
    "XLF": "XLF",
    "XLE": "XLE",
    "LLY": "LLY",
    "BTC-USD": "BTCUSD",
}

_mt5_inicializado = False


def inicializar_llm():
    """Conecta con Groq (nube) si existe API key, o con Ollama (local) si no."""
    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq

            print("  [+] Modo LLM: Groq Cloud API (Llama-3.3-70b)")
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=GROQ_API_KEY,
                temperature=0.2,
            )
        except Exception as e:
            print(f"  [!] Error al inicializar Groq: {e}. Intentando Ollama...")

    try:
        from langchain_ollama import ChatOllama

        print("  [+] Modo LLM: Local Ollama (Qwen2.5)")
        return ChatOllama(model="qwen2.5", temperature=0.2)
    except Exception as e:
        print(f"  [!] Advertencia: No se pudo cargar ningún LLM ({e}). Se omitirá el resumen IA.")
        return None


def calcular_rsi(data, window=14):
    """Calcula el RSI con suavizado de Wilder (compatible con TradingView)."""
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    avg_loss_safe = avg_loss.replace(0, float("nan"))
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def extraer_pivots(series, kind, window=2):
    """Extrae pivots locales recientes de una serie de precios."""
    valores = []
    for idx in range(window, len(series) - window):
        centro = series.iloc[idx]
        if pd.isna(centro):
            continue
        izquierda = series.iloc[idx - window : idx]
        derecha = series.iloc[idx + 1 : idx + window + 1]
        if kind == "high" and centro >= izquierda.max() and centro >= derecha.max():
            valores.append(float(centro))
        elif kind == "low" and centro <= izquierda.min() and centro <= derecha.min():
            valores.append(float(centro))
    return valores


def analizar_estructura_precio(df):
    """Detecta higher highs/lows y lower highs/lows usando pivots recientes."""
    pivot_highs = extraer_pivots(df["High"], kind="high")
    pivot_lows = extraer_pivots(df["Low"], kind="low")

    hh = len(pivot_highs) >= 2 and pivot_highs[-1] > pivot_highs[-2]
    lh = len(pivot_highs) >= 2 and pivot_highs[-1] < pivot_highs[-2]
    hl = len(pivot_lows) >= 2 and pivot_lows[-1] > pivot_lows[-2]
    ll = len(pivot_lows) >= 2 and pivot_lows[-1] < pivot_lows[-2]

    if hh and hl:
        estructura = "ALCISTA"
    elif lh and ll:
        estructura = "BAJISTA"
    elif hh or hl:
        estructura = "ALCISTA PARCIAL"
    elif lh or ll:
        estructura = "BAJISTA PARCIAL"
    else:
        estructura = "NEUTRA"

    etiquetas = []
    if hh:
        etiquetas.append("Higher Highs")
    if hl:
        etiquetas.append("Higher Lows")
    if lh:
        etiquetas.append("Lower Highs")
    if ll:
        etiquetas.append("Lower Lows")

    return {
        "estructura": estructura,
        "hh": hh,
        "hl": hl,
        "lh": lh,
        "ll": ll,
        "texto": ", ".join(etiquetas) if etiquetas else "Sin patrón claro",
    }


def evaluar_senal(precio_actual, rsi_actual, sma_200_actual, estructura):
    """Evalúa la señal combinando RSI, tendencia macro y estructura."""
    tendencia_alcista = precio_actual > sma_200_actual
    texto_tendencia = "ALCISTA 🟢" if tendencia_alcista else "BAJISTA 🔴"

    if rsi_actual <= RSI_OVERSOLD and tendencia_alcista and estructura["hh"] and estructura["hl"]:
        return {
            "estado": "🟢 COMPRA FUERTE (Higher Highs + Higher Lows en Tendencia Alcista)",
            "texto_tendencia": texto_tendencia,
            "enviar_alerta": True,
        }
    if rsi_actual <= RSI_OVERSOLD and tendencia_alcista and estructura["hl"]:
        return {
            "estado": "🟢 COMPRA CONFIRMADA (Sobreventa dentro de Tendencia Alcista)",
            "texto_tendencia": texto_tendencia,
            "enviar_alerta": True,
        }
    if rsi_actual >= RSI_OVERBOUGHT and (not tendencia_alcista) and estructura["lh"] and estructura["ll"]:
        return {
            "estado": "🔴 VENTA FUERTE (Lower Highs + Lower Lows en Tendencia Bajista)",
            "texto_tendencia": texto_tendencia,
            "enviar_alerta": True,
        }
    if rsi_actual >= RSI_OVERBOUGHT and (not tendencia_alcista) and estructura["lh"]:
        return {
            "estado": "🔴 VENTA/CORRECCIÓN (Sobrecompra dentro de Tendencia Bajista)",
            "texto_tendencia": texto_tendencia,
            "enviar_alerta": True,
        }
    if rsi_actual <= RSI_OVERSOLD and tendencia_alcista and not estructura["hl"]:
        motivo = "Falta confirmación de higher lows para validar compra."
    elif rsi_actual >= RSI_OVERBOUGHT and (not tendencia_alcista) and not estructura["lh"]:
        motivo = "Falta confirmación de lower highs para validar venta."
    elif rsi_actual <= RSI_OVERSOLD and not tendencia_alcista:
        motivo = "RSI en sobreventa descartado por tendencia macro bajista."
    elif rsi_actual >= RSI_OVERBOUGHT and tendencia_alcista:
        motivo = "RSI en sobrecompra descartado por fuerza de tendencia alcista."
    else:
        motivo = f"RSI neutral ({rsi_actual}). Sin señal operativa."

    return {
        "estado": "",
        "texto_tendencia": texto_tendencia,
        "enviar_alerta": False,
        "motivo": motivo,
    }


def buscar_noticias_recientes(query, max_results=3):
    """Busca las últimas noticias en internet usando DDGS."""
    texto_noticias = ""
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=max_results, timelimit="w")
            for idx, r in enumerate(results, 1):
                texto_noticias += f"\n{idx}. {r.get('title')} - {r.get('body')}"
    except Exception as e:
        print(f"  [!] Error buscando noticias para '{query}': {e}")
        texto_noticias = "No se pudieron obtener noticias recientes."
    return texto_noticias


def construir_url_tradingview(ticker_symbol, tipo):
    """Genera la URL para abrir el gráfico en TradingView."""
    symbol = TRADINGVIEW_SYMBOLS.get(ticker_symbol)
    if not symbol:
        if tipo == "Cripto":
            base = ticker_symbol.replace("-USD", "").upper()
            symbol = f"BINANCE:{base}USDT"
        elif tipo == "ETF":
            symbol = f"AMEX:{ticker_symbol}"
        else:
            symbol = f"NASDAQ:{ticker_symbol}"
    return f"https://es.tradingview.com/chart/?symbol={symbol}"


def construir_simbolo_metatrader(ticker_symbol):
    """Devuelve el símbolo MT5 (con sufijo de broker opcional)."""
    base = METATRADER_SYMBOLS.get(ticker_symbol, ticker_symbol.replace("-USD", "USD"))
    return f"{base}{MT5_SUFFIX}" if MT5_SUFFIX else base


def inicializar_metatrader():
    """Conecta con la terminal MetaTrader 5 local."""
    global _mt5_inicializado
    if _mt5_inicializado:
        return True

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("  [!] MetaTrader5 no instalado. Usa: pip install MetaTrader5")
        return False

    kwargs = {"path": MT5_PATH} if MT5_PATH else {}
    if not mt5.initialize(**kwargs):
        print(f"  [!] No se pudo conectar a MT5: {mt5.last_error()}")
        return False

    _mt5_inicializado = True
    print("  [+] Conectado a MetaTrader 5")
    return True


def obtener_datos_yfinance(ticker_symbol):
    """Obtiene histórico desde Yahoo Finance (modo standalone)."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1y")
        if df.empty or len(df) < 200:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"  [!] Error Yahoo Finance para {ticker_symbol}: {e}")
        return None


def obtener_datos_metatrader(ticker_symbol, tipo):
    """Obtiene histórico diario desde MetaTrader 5."""
    if not inicializar_metatrader():
        return None

    import MetaTrader5 as mt5

    simbolo = construir_simbolo_metatrader(ticker_symbol)
    if not mt5.symbol_select(simbolo, True):
        alternativas = [simbolo]
        if tipo == "Cripto":
            alternativas.append(f"{simbolo}m")
        if tipo in ("Acción", "ETF"):
            alternativas.extend([f"#{simbolo}", f"{simbolo}.US"])

        simbolo = next((s for s in alternativas if mt5.symbol_select(s, True)), None)
        if not simbolo:
            print(f"  [!] Símbolo MT5 no encontrado para {ticker_symbol}")
            return None

    rates = mt5.copy_rates_from_pos(simbolo, mt5.TIMEFRAME_D1, 0, 300)
    if rates is None or len(rates) < 200:
        print(f"  [!] Datos MT5 insuficientes para {simbolo}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def obtener_datos_historicos(ticker_symbol, tipo):
    """Obtiene datos según la fuente configurada."""
    if FUENTE_DATOS == "yfinance":
        df = obtener_datos_yfinance(ticker_symbol)
        return df, "Yahoo Finance (standalone)" if df is not None else (None, None)

    if FUENTE_DATOS == "metatrader":
        df = obtener_datos_metatrader(ticker_symbol, tipo)
        return df, "MetaTrader 5" if df is not None else (None, None)

    df = obtener_datos_metatrader(ticker_symbol, tipo)
    if df is not None:
        return df, "MetaTrader 5"

    df = obtener_datos_yfinance(ticker_symbol)
    if df is not None:
        return df, "Yahoo Finance (standalone)"

    return None, None


def construir_bloque_plataformas(ticker_symbol, tipo, fuente_datos):
    """Genera enlaces e instrucciones para TradingView, MetaTrader y modo standalone."""
    url_tv = construir_url_tradingview(ticker_symbol, tipo)
    simbolo_mt5 = construir_simbolo_metatrader(ticker_symbol)

    return f"""
📊 <b>Modo standalone:</b> análisis completo arriba (fuente: {html.escape(fuente_datos)})
📈 <b>TradingView:</b> <a href="{url_tv}">Abrir gráfico</a>
💻 <b>MetaTrader 5:</b> busca <b>{html.escape(simbolo_mt5)}</b> en Market Watch
<i>Si no aparece, prueba variantes como #{html.escape(simbolo_mt5)} o {html.escape(simbolo_mt5)}m según tu broker.</i>
    """.strip()


def telegram_configurado():
    return (
        TELEGRAM_TOKEN
        and TELEGRAM_TOKEN != "TU_TOKEN_DE_TELEGRAM_LOCAL"
        and CHAT_ID
        and CHAT_ID != "TU_CHAT_ID_LOCAL"
    )


def enviar_telegram(mensaje):
    """Envía la notificación formateada a Telegram."""
    if not telegram_configurado():
        print("  [!] Telegram no configurado. Se muestra el mensaje en consola:\n")
        print(mensaje)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": mensaje,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    try:
        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        print("  [✓] Alerta enviada con éxito a Telegram.")
    except Exception as e:
        print(f"  [!] Error enviando mensaje a Telegram: {e}")


def enviar_mensaje_prueba():
    """Envía un mensaje simple para verificar Telegram sin depender de señales."""
    mensaje = """
🧪 <b>PRUEBA DE TELEGRAM</b>

El bot está conectado correctamente y puede enviarte mensajes.
Si estás viendo esto, la configuración de Telegram funciona.
    """
    enviar_telegram(mensaje.strip())


def preparar_dataframe_analisis(df):
    """Añade indicadores base al dataframe."""
    df = df.copy()
    df["RSI"] = calcular_rsi(df, window=14)
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    return df


def obtener_ultimo_valor_valido(series):
    """Devuelve el último valor no nulo de una serie."""
    serie_valida = series.dropna()
    if serie_valida.empty:
        return None
    return float(serie_valida.iloc[-1])


def calcular_rendimiento_neto(precio_entrada, precio_salida):
    """Aplica spread y comisiones a una operación larga."""
    entrada_efectiva = precio_entrada * (1 + BACKTEST_SPREAD_PCT / 2) * (1 + BACKTEST_COMMISSION_PCT)
    salida_efectiva = precio_salida * (1 - BACKTEST_SPREAD_PCT / 2) * (1 - BACKTEST_COMMISSION_PCT)
    return (salida_efectiva / entrada_efectiva) - 1


def exportar_resultados_backtest(resultados):
    """Exporta resumen y trades a CSV."""
    os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

    resumen_rows = []
    trades_rows = []
    for resultado in resultados:
        resumen_rows.append(
            {
                "ticker": resultado["ticker"],
                "nombre": resultado["nombre"],
                "fuente_datos": resultado["fuente_datos"],
                "capital_final": resultado["capital_final"],
                "rentabilidad_total": resultado["rentabilidad_total"],
                "numero_trades": resultado["numero_trades"],
                "win_rate": resultado["win_rate"],
                "ganadoras": resultado["ganadoras"],
                "perdedoras": resultado["perdedoras"],
            }
        )
        for trade in resultado["trades"]:
            trades_rows.append(
                {
                    "ticker": resultado["ticker"],
                    "nombre": resultado["nombre"],
                    "entrada": trade["entrada"],
                    "salida": trade["salida"],
                    "precio_entrada": trade["precio_entrada"],
                    "precio_salida": trade["precio_salida"],
                    "rendimiento_pct": round(trade["rendimiento"] * 100, 2),
                    "motivo_salida": trade["estado_salida"],
                }
            )

    resumen_path = os.path.join(BACKTEST_OUTPUT_DIR, "resumen_backtest.csv")
    trades_path = os.path.join(BACKTEST_OUTPUT_DIR, "trades_backtest.csv")
    pd.DataFrame(resumen_rows).to_csv(resumen_path, index=False)
    pd.DataFrame(trades_rows).to_csv(trades_path, index=False)
    return resumen_path, trades_path


def evaluar_activo(activo, llm):
    ticker_symbol = activo["ticker"]
    nombre = activo["nombre"]
    tipo = activo["tipo"]

    print("\n--------------------------------------------------")
    print(f"Analizando [{tipo}] {nombre} ({ticker_symbol})...")

    df, fuente_datos = obtener_datos_historicos(ticker_symbol, tipo)
    if df is None:
        print(f"  [-] No se pudieron obtener datos para {ticker_symbol}.")
        return

    print(f"  [i] Fuente de datos: {fuente_datos}")

    df = preparar_dataframe_analisis(df)
    estructura = analizar_estructura_precio(df)

    precio_actual = round(float(df["Close"].iloc[-1]), 2)
    rsi_valido = obtener_ultimo_valor_valido(df["RSI"])
    sma_200_valido = obtener_ultimo_valor_valido(df["SMA_200"])

    if rsi_valido is None or sma_200_valido is None:
        print(f"  [-] Indicadores incompletos para {ticker_symbol}.")
        return

    rsi_actual = round(rsi_valido, 2)
    sma_200_actual = round(sma_200_valido, 2)

    senal = evaluar_senal(precio_actual, rsi_actual, sma_200_actual, estructura)
    texto_tendencia = senal["texto_tendencia"]

    print(
        f"  Precio: ${precio_actual} | SMA 200: ${sma_200_actual} ({texto_tendencia}) | RSI: {rsi_actual}"
    )
    print(f"  Estructura: {estructura['estructura']} | {estructura['texto']}")

    if not senal["enviar_alerta"]:
        print(f"  [-] Filtro: {senal['motivo']}")
        return

    estado = senal["estado"]
    print(f"  [+] SEÑAL DETECTADA: {estado}")

    query_busqueda = f"{nombre} {tipo} market news"
    noticias = buscar_noticias_recientes(query_busqueda, max_results=3)

    analisis_ia = "Análisis IA no disponible."
    if llm:
        print("  [+] Generando síntesis con IA...")
        prompt = f"""
        Eres un analista financiero experto.
        El activo {nombre} ({ticker_symbol}), de tipo {tipo}, presenta las siguientes métricas:
        - Precio Actual: ${precio_actual}
        - Tendencia Macro (SMA 200): ${sma_200_actual} ({texto_tendencia})
        - RSI (14): {rsi_actual}
        - Estructura de precio: {estructura["estructura"]} ({estructura["texto"]})
        - Diagnóstico técnico: {estado}

        Noticias recientes sobre {nombre}:
        {noticias}

        Proporciona un análisis ejecutivo unificado (técnico + fundamental) en máximo 3 frases.
        Indica si las noticias justifican la entrada y qué nivel o soporte vigilar.
        """
        try:
            res = llm.invoke(prompt)
            analisis_ia = res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            print(f"  [!] Error invocando la IA: {e}")
            analisis_ia = "No se pudo procesar el análisis de la IA."

    bloque_plataformas = construir_bloque_plataformas(ticker_symbol, tipo, fuente_datos)
    analisis_ia_seguro = html.escape(analisis_ia)

    mensaje_telegram = f"""
🚨 <b>ALERTA DE TRADING [{tipo.upper()}]: {nombre} ({ticker_symbol})</b>

📌 <b>Diagnóstico:</b> {estado}
💵 <b>Precio Actual:</b> ${precio_actual}
📈 <b>SMA 200 (Macro):</b> ${sma_200_actual} ({texto_tendencia})
📊 <b>RSI (14):</b> {rsi_actual}
🧭 <b>Estructura:</b> {html.escape(estructura["estructura"])} ({html.escape(estructura["texto"])})

💡 <b>Análisis IA + Noticias:</b>
{analisis_ia_seguro}

📲 <b>PLATAFORMAS:</b>
{bloque_plataformas}
    """

    enviar_telegram(mensaje_telegram.strip())


def backtest_activo(activo, capital_inicial=BACKTEST_INITIAL_CAPITAL):
    """Simula entradas y salidas sobre histórico diario."""
    ticker_symbol = activo["ticker"]
    nombre = activo["nombre"]
    tipo = activo["tipo"]

    df, fuente_datos = obtener_datos_historicos(ticker_symbol, tipo)
    if df is None:
        print(f"  [-] Backtest omitido para {ticker_symbol}: sin datos.")
        return None

    df = preparar_dataframe_analisis(df)
    capital = capital_inicial
    en_posicion = False
    precio_entrada = 0.0
    fecha_entrada = None
    stop_loss = None
    take_profit = None
    trades = []

    for idx in range(200, len(df)):
        ventana = df.iloc[: idx + 1].copy()
        precio_actual = float(ventana["Close"].iloc[-1])
        rsi_actual = ventana["RSI"].iloc[-1]
        sma_200_actual = ventana["SMA_200"].iloc[-1]

        if pd.isna(rsi_actual) or pd.isna(sma_200_actual):
            continue

        estructura = analizar_estructura_precio(ventana)
        senal = evaluar_senal(precio_actual, float(rsi_actual), float(sma_200_actual), estructura)
        estado = senal["estado"]

        es_compra = estado.startswith("🟢")
        es_venta = estado.startswith("🔴")

        if not en_posicion and es_compra:
            en_posicion = True
            precio_entrada = precio_actual
            fecha_entrada = ventana.index[-1]
            stop_loss = precio_entrada * (1 - BACKTEST_STOP_LOSS_PCT)
            take_profit = precio_entrada * (1 + BACKTEST_TAKE_PROFIT_PCT)
        elif en_posicion:
            high_actual = float(ventana["High"].iloc[-1])
            low_actual = float(ventana["Low"].iloc[-1])
            precio_salida = None
            estado_salida = None

            if low_actual <= stop_loss:
                precio_salida = stop_loss
                estado_salida = "Stop Loss"
            elif high_actual >= take_profit:
                precio_salida = take_profit
                estado_salida = "Take Profit"
            elif es_venta:
                precio_salida = precio_actual
                estado_salida = estado

            if precio_salida is None:
                continue

            rendimiento = calcular_rendimiento_neto(precio_entrada, precio_salida)
            capital *= 1 + rendimiento
            trades.append(
                {
                    "entrada": fecha_entrada,
                    "salida": ventana.index[-1],
                    "precio_entrada": precio_entrada,
                    "precio_salida": precio_salida,
                    "rendimiento": rendimiento,
                    "estado_salida": estado_salida,
                }
            )
            en_posicion = False
            precio_entrada = 0.0
            fecha_entrada = None
            stop_loss = None
            take_profit = None

    if en_posicion:
        precio_final = float(df["Close"].iloc[-1])
        rendimiento = calcular_rendimiento_neto(precio_entrada, precio_final)
        capital *= 1 + rendimiento
        trades.append(
            {
                "entrada": fecha_entrada,
                "salida": df.index[-1],
                "precio_entrada": precio_entrada,
                "precio_salida": precio_final,
                "rendimiento": rendimiento,
                "estado_salida": "Cierre al final del backtest",
            }
        )

    ganancias = [t for t in trades if t["rendimiento"] > 0]
    perdidas = [t for t in trades if t["rendimiento"] <= 0]
    rentabilidad_total = ((capital / capital_inicial) - 1) * 100
    win_rate = (len(ganancias) / len(trades) * 100) if trades else 0.0

    return {
        "ticker": ticker_symbol,
        "nombre": nombre,
        "fuente_datos": fuente_datos,
        "trades": trades,
        "capital_final": round(capital, 2),
        "rentabilidad_total": round(rentabilidad_total, 2),
        "numero_trades": len(trades),
        "win_rate": round(win_rate, 2),
        "ganadoras": len(ganancias),
        "perdedoras": len(perdidas),
    }


def ejecutar_backtest():
    """Ejecuta un backtest simple sobre todo el portafolio."""
    print("=== INICIANDO BACKTEST DEL PORTAFOLIO ===")
    print(f"Fuente de datos configurada: {FUENTE_DATOS}")
    print(
        "Parámetros: "
        f"capital=${BACKTEST_INITIAL_CAPITAL}, "
        f"comisión={BACKTEST_COMMISSION_PCT * 100}%, "
        f"spread={BACKTEST_SPREAD_PCT * 100}%, "
        f"SL={BACKTEST_STOP_LOSS_PCT * 100}%, "
        f"TP={BACKTEST_TAKE_PROFIT_PCT * 100}%"
    )

    resultados = []
    for activo in PORTAFOLIO:
        print(f"\nBacktesting {activo['nombre']} ({activo['ticker']})...")
        resultado = backtest_activo(activo)
        if not resultado:
            continue

        resultados.append(resultado)
        print(
            "  "
            f"Trades: {resultado['numero_trades']} | "
            f"Win rate: {resultado['win_rate']}% | "
            f"Rentabilidad: {resultado['rentabilidad_total']}% | "
            f"Capital final: ${resultado['capital_final']}"
        )

    if not resultados:
        print("\n=== BACKTEST SIN RESULTADOS ===")
        return

    rentabilidad_media = sum(r["rentabilidad_total"] for r in resultados) / len(resultados)
    total_trades = sum(r["numero_trades"] for r in resultados)
    mejor = max(resultados, key=lambda r: r["rentabilidad_total"])
    peor = min(resultados, key=lambda r: r["rentabilidad_total"])

    print("\n=== RESUMEN BACKTEST ===")
    print(f"Activos analizados: {len(resultados)}")
    print(f"Trades totales: {total_trades}")
    print(f"Rentabilidad media por activo: {round(rentabilidad_media, 2)}%")
    print(
        f"Mejor activo: {mejor['ticker']} ({mejor['rentabilidad_total']}%) | "
        f"Peor activo: {peor['ticker']} ({peor['rentabilidad_total']}%)"
    )
    resumen_path, trades_path = exportar_resultados_backtest(resultados)
    print(f"CSV resumen: {resumen_path}")
    print(f"CSV trades: {trades_path}")


def ejecutar_escaneo():
    print("=== INICIANDO ESCANEO DE PORTAFOLIO MULTIACTIVO ===")
    print(f"Fuente de datos configurada: {FUENTE_DATOS}")

    llm = inicializar_llm()

    for activo in PORTAFOLIO:
        evaluar_activo(activo, llm)
        time.sleep(2)

    print("\n=== ESCANEO FINALIZADO ===")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "backtest":
        ejecutar_backtest()
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "test-telegram":
        enviar_mensaje_prueba()
    else:
        ejecutar_escaneo()
