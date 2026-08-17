import html
import os
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

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


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

    df["RSI"] = calcular_rsi(df, window=14)
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    precio_actual = round(float(df["Close"].iloc[-1]), 2)
    rsi_actual = round(float(df["RSI"].iloc[-1]), 2)
    sma_200_actual = round(float(df["SMA_200"].iloc[-1]), 2)

    if pd.isna(rsi_actual) or pd.isna(sma_200_actual):
        print(f"  [-] Indicadores incompletos para {ticker_symbol}.")
        return

    tendencia_alcista = precio_actual > sma_200_actual
    texto_tendencia = "ALCISTA 🟢" if tendencia_alcista else "BAJISTA 🔴"

    print(
        f"  Precio: ${precio_actual} | SMA 200: ${sma_200_actual} ({texto_tendencia}) | RSI: {rsi_actual}"
    )

    estado = ""
    if rsi_actual <= RSI_OVERSOLD and tendencia_alcista:
        estado = "🟢 COMPRA CONFIRMADA (Sobreventa dentro de Tendencia Alcista)"
    elif rsi_actual >= RSI_OVERBOUGHT and not tendencia_alcista:
        estado = "🔴 VENTA/CORRECCIÓN (Sobrecompra dentro de Tendencia Bajista)"
    elif rsi_actual <= RSI_OVERSOLD and not tendencia_alcista:
        print("  [-] Filtro: RSI en sobreventa descartado por tendencia macro bajista.")
        return
    elif rsi_actual >= RSI_OVERBOUGHT and tendencia_alcista:
        print("  [-] Filtro: RSI en sobrecompra descartado por fuerza de tendencia alcista.")
        return
    else:
        print(f"  [-] Filtro: RSI neutral ({rsi_actual}). Sin señal operativa.")
        return

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

💡 <b>Análisis IA + Noticias:</b>
{analisis_ia_seguro}

📲 <b>PLATAFORMAS:</b>
{bloque_plataformas}
    """

    enviar_telegram(mensaje_telegram.strip())


def ejecutar_escaneo():
    print("=== INICIANDO ESCANEO DE PORTAFOLIO MULTIACTIVO ===")
    print(f"Fuente de datos configurada: {FUENTE_DATOS}")

    llm = inicializar_llm()

    for activo in PORTAFOLIO:
        evaluar_activo(activo, llm)
        time.sleep(2)

    print("\n=== ESCANEO FINALIZADO ===")


if __name__ == "__main__":
    ejecutar_escaneo()
