import numpy as np
import pandas as pd
import logging
import asyncio
import os
import traceback
import time
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import json
from typing import List, Dict, Optional, Union, Tuple

# Comprehensive warning and compatibility management
import os
import sys
import warnings
import logging

# Suppress all warnings early and aggressively
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Silence TensorFlow and Keras logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)

# Advanced warning suppression
def noop_warn(*args, **kwargs):
    pass

# Monkey patch warning mechanisms
warnings.warn = noop_warn
warnings.showwarning = noop_warn

# Import TensorFlow with full compatibility configuration
import tensorflow as tf
import keras

# Compatibility and warning suppression
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Disable v2 behavior and eager execution
try:
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass

# Disable Keras interactive logging
keras.utils.disable_interactive_logging()

# Import Keras and TensorFlow modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure logging for your application
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lstm_model')
logger.setLevel(logging.INFO)

# Log versions with minimal output
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Keras version: {keras.__version__}")
logger.info(f"Python version: {sys.version}")

# Explicitly use compatibility layer loss functions
losses = tf.compat.v1.losses
sparse_softmax_cross_entropy = losses.sparse_softmax_cross_entropy

# Additional warning suppression for Keras
try:
    import tensorflow.keras.utils
    tensorflow.keras.utils.warn = noop_warn
except Exception:
    pass

# تهيئة OpenAI
import openai
openai.api_key = "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"

# تهيئة MT5
import MetaTrader5 as mt5

def initialize_mt5(symbol='EURUSD'):
    """
    Robust MetaTrader5 initialization with flexible error handling
    
    :param symbol: Default symbol to validate
    :return: Boolean indicating successful initialization
    """
    try:
        # Attempt to initialize MT5
        if not mt5.initialize():
            logger.error("Failed to initialize MetaTrader5")
            return False
        
        # Validate symbol availability
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Symbol {symbol} is not available. Please add it to Market Watch in MT5.")
            return False
        
        logger.info(f"MetaTrader5 initialized successfully with symbol {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"MetaTrader5 initialization error: {e}")
        return False

# Global MT5 initialization flag
MT5_INITIALIZED = initialize_mt5()

# Modify main function to handle MT5 initialization
def main():
    try:
        # Check MT5 initialization
        if not MT5_INITIALIZED:
            logger.error("MetaTrader5 is not initialized. Cannot start trading.")
            return
        
        # Configuration
        initial_balance = 10000  # USD
        trading_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']  # Multi-symbol support
        
        # Start multi-symbol trading
        multi_symbol_trading(trading_symbols, initial_balance)
    
    except Exception as e:
        logger.error(f"Critical error in main trading loop: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Ensure MT5 connection is properly closed
        if MT5_INITIALIZED:
            mt5.shutdown()

# التحقق من أن الرمز متوفر
symbol = "EURUSD"  # تغيير الرمز إلى EURUSD
try:
    if not mt5.symbol_select(symbol, True):
        raise Exception(f"Symbol {symbol} is not available. Please add it to Market Watch in MT5.")
except Exception as e:
    print(f"An error occurred: {e}")
    mt5.shutdown()
    quit()

# دالة لترجمة النص من الإنجليزية إلى العربية
from deep_translator import GoogleTranslator  # استيراد مكتبة الترجمة
def translate_to_arabic(text):
    try:
        translated = GoogleTranslator(source='auto', target='ar').translate(text)
        return translated
    except Exception as e:
        print(f"Failed to translate: {e}")
        return text  # إذا فشلت الترجمة، يتم إرجاع النص الأصلي

async def get_candlestick_data(symbol, timeframe, count):
    try:
        # جلب البيانات من MT5
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None:
            raise Exception(f"Failed to retrieve data for {symbol}. Please check if the symbol is correct and MT5 is connected.")
        
        # إنشاء DataFrame يدويًا مع تسمية الأعمدة
        df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
        
        # تحويل الطابع الزمني إلى تاريخ قابل للقراءة
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        return df
    except Exception as e:
        print(f"Error in get_candlestick_data: {e}")

def compute_rsi(prices: Union[pd.Series, np.ndarray, List[float]], window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) with robust error handling
    
    :param prices: Input price series
    :param window: RSI calculation window
    :return: RSI series with at least 100 elements
    """
    try:
        # Convert input to pandas Series if not already
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Ensure minimum length for computation
        if len(prices) < window:
            logger.warning(f"Insufficient data for RSI computation. Required {window}, got {len(prices)}")
            # Return a series of zeros if not enough data
            return pd.Series(np.zeros(100), index=range(100))
        
        # Compute price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Compute average gain and loss over the window
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # Compute relative strength
        rs = avg_gain / avg_loss
        
        # Compute RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Pad the series to ensure 100 elements
        if len(rsi) < 100:
            padding = np.zeros(100 - len(rsi))
            rsi = pd.Series(np.concatenate([padding, rsi.values]), index=range(100))
        elif len(rsi) > 100:
            rsi = rsi.tail(100)
        
        # Ensure the series has 100 elements and a range index
        rsi.index = range(100)
        
        # Log RSI computation details
        logger.debug(f"RSI Computation: length={len(rsi)}, values={rsi.values}")
        
        return rsi
    
    except Exception as e:
        logger.error(f"Error computing RSI: {e}")
        # Return a series of zeros in case of any computation error
        return pd.Series(np.zeros(100), index=range(100))

def compute_macd(prices: Union[pd.Series, np.ndarray, List[float]], 
              short_window: int = 12, 
              long_window: int = 26, 
              signal_window: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Moving Average Convergence Divergence (MACD)
    
    :param prices: Price series
    :param short_window: Short-term moving average window
    :param long_window: Long-term moving average window
    :param signal_window: Signal line window
    :return: Tuple of MACD line, signal line, and histogram
    """
    try:
        # Convert input to pandas Series
        if isinstance(prices, list):
            prices = pd.Series(prices)
        elif isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        
        # Log input details for debugging
        logger.debug(f"MACD Input: type={type(prices)}, length={len(prices)}, windows=[{short_window}, {long_window}, {signal_window}]")
        
        # Validate input
        if len(prices) < max(short_window, long_window, signal_window):
            logger.warning(f"Insufficient data for MACD computation. Need at least {max(short_window, long_window, signal_window)} points, got {len(prices)}")
            return (
                np.zeros(100), 
                np.zeros(100), 
                np.zeros(100)
            )
        
        # Compute exponential moving averages
        exp_short = prices.ewm(span=short_window, adjust=False).mean()
        exp_long = prices.ewm(span=long_window, adjust=False).mean()
        
        # Compute MACD line
        macd_line = exp_short - exp_long
        
        # Compute signal line
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        
        # Compute histogram
        histogram = macd_line - signal_line
        
        # Ensure consistent return values
        def _safe_to_array(series, length=100):
            arr = series.values
            if len(arr) < length:
                arr = np.pad(arr, (length - len(arr), 0), mode='constant')
            return arr[:length]
        
        # Log output details for debugging
        result = (
            _safe_to_array(macd_line), 
            _safe_to_array(signal_line), 
            _safe_to_array(histogram)
        )
        
        logger.debug(f"MACD Output: lengths={[len(arr) for arr in result]}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in MACD computation: {e}")
        
        # Return zero arrays of consistent length
        return (
            np.zeros(100), 
            np.zeros(100), 
            np.zeros(100)
        )

def compute_bollinger_bands(prices: Union[pd.Series, np.ndarray, List[float]], window: int = 20, num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Bollinger Bands with robust error handling
    
    :param prices: Input price series
    :param window: Rolling window for computation
    :param num_std: Number of standard deviations
    :return: Tuple of (upper_band, middle_band, lower_band) with 100 elements
    """
    try:
        # Convert input to pandas Series if not already
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Ensure minimum length for computation
        if len(prices) < window:
            logger.warning(f"Insufficient data for Bollinger Bands computation. Required {window}, got {len(prices)}")
            # Return zero arrays if not enough data
            return (
                np.zeros(100), 
                np.zeros(100), 
                np.zeros(100)
            )
        
        # Compute rolling mean and standard deviation
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        
        # Compute Bollinger Bands
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Pad or truncate to ensure 100 elements
        def _pad_or_truncate(series):
            if len(series) < 100:
                padding = np.zeros(100 - len(series))
                return np.concatenate([padding, series.values])
            elif len(series) > 100:
                return series.tail(100).values
            return series.values
        
        # Ensure consistent output
        upper_band_arr = _pad_or_truncate(upper_band)
        middle_band_arr = _pad_or_truncate(rolling_mean)
        lower_band_arr = _pad_or_truncate(lower_band)
        
        # Log Bollinger Bands computation details
        logger.debug(f"Bollinger Bands Computation: lengths={[len(arr) for arr in (upper_band_arr, middle_band_arr, lower_band_arr)]}")
        
        return (upper_band_arr, middle_band_arr, lower_band_arr)
    
    except Exception as e:
        logger.error(f"Error computing Bollinger Bands: {e}")
        # Return zero arrays in case of any computation error
        return (
            np.zeros(100), 
            np.zeros(100), 
            np.zeros(100)
        )

def compute_adx(high, low, close, period=14):
    try:
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        return adx
    except Exception as e:
        print(f"Error in compute_adx: {e}")

def compute_momentum(prices, period=14):
    try:
        return prices.diff(period)
    except Exception as e:
        print(f"Error in compute_momentum: {e}")

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    try:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    except Exception as e:
        print(f"Error in compute_stochastic: {e}")

def compute_atr(high: Union[pd.Series, np.ndarray, List[float]], 
              low: Union[pd.Series, np.ndarray, List[float]], 
              close: Union[pd.Series, np.ndarray, List[float]], 
              window: int = 14) -> np.ndarray:
    """
    Compute Average True Range (ATR) with robust error handling
    
    :param high: High price series
    :param low: Low price series
    :param close: Closing price series
    :param window: ATR calculation window
    :return: ATR series with 100 elements
    """
    try:
        # Convert inputs to pandas Series if not already
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        
        # Ensure consistent length of input series
        min_length = min(len(high), len(low), len(close))
        
        # Truncate series to minimum length
        high = high.head(min_length)
        low = low.head(min_length)
        close = close.head(min_length)
        
        # Ensure minimum length for computation
        if min_length < window:
            logger.warning(f"Insufficient data for ATR computation. Required {window}, got {min_length}")
            # Return zero array if not enough data
            return np.zeros(100)
        
        # Compute True Range
        def true_range(high, low, prev_close):
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            return max(tr1, tr2, tr3)
        
        # Compute true range series
        true_ranges = [true_range(high.iloc[i], low.iloc[i], close.iloc[i-1]) 
                       for i in range(1, len(high))]
        true_ranges.insert(0, high.iloc[0] - low.iloc[0])  # First TR is H-L
        true_ranges = pd.Series(true_ranges)
        
        # Compute ATR using Wilder's smoothing method
        atr = true_ranges.ewm(span=window, adjust=False).mean()
        
        # Pad or truncate to ensure 100 elements
        if len(atr) < 100:
            padding = np.zeros(100 - len(atr))
            atr_arr = np.concatenate([padding, atr.values])
        elif len(atr) > 100:
            atr_arr = atr.tail(100).values
        else:
            atr_arr = atr.values
        
        # Log ATR computation details
        logger.debug(f"ATR Computation: length={len(atr_arr)}, values={atr_arr}")
        
        return atr_arr
    
    except Exception as e:
        logger.error(f"Error computing ATR: {e}")
        # Return zero array in case of any computation error
        return np.zeros(100)

def compute_cci(high, low, close, period=20):
    try:
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    except Exception as e:
        print(f"Error in compute_cci: {e}")

def compute_mfi(high, low, close, volume, period=14):
    try:
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi
    except Exception as e:
        print(f"Error in compute_mfi: {e}")

def compute_williams_r(high, low, close, period=14):
    try:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    except Exception as e:
        print(f"Error in compute_williams_r: {e}")

def compute_vwap(high, low, close, volume):
    try:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    except Exception as e:
        print(f"Error in compute_vwap: {e}")

def compute_supertrend(high, low, close, period=10, multiplier=3):
    try:
        atr = compute_atr(high, low, close, period)
        basic_upper = ((high + low) / 2) + (multiplier * atr)
        basic_lower = ((high + low) / 2) - (multiplier * atr)
        supertrend = []
        for i in range(len(close)):
            if i == 0:
                supertrend.append(basic_upper.iloc[i])
            else:
                if close.iloc[i] > supertrend[i-1]:
                    supertrend.append(max(basic_lower.iloc[i], supertrend[i-1]))
                else:
                    supertrend.append(min(basic_upper.iloc[i], supertrend[i-1]))
        return pd.Series(supertrend, index=close.index)
    except Exception as e:
        print(f"Error in compute_supertrend: {e}")

def compute_parabolic_sar(high, low, close, acceleration=0.02, maximum=0.2):
    try:
        sar = []
        ep = 0
        af = acceleration
        trend = 1  # 1 for uptrend, -1 for downtrend
        for i in range(len(close)):
            if i == 0:
                sar.append(close.iloc[i])
                ep = high.iloc[i] if trend == 1 else low.iloc[i]
            else:
                sar.append(sar[i-1] + af * (ep - sar[i-1]))
                if trend == 1:
                    if low.iloc[i] < sar[i]:
                        trend = -1
                        sar[i] = ep
                        ep = low.iloc[i]
                        af = acceleration
                    else:
                        if high.iloc[i] > ep:
                            ep = high.iloc[i]
                            af = min(af + acceleration, maximum)
                else:
                    if high.iloc[i] > sar[i]:
                        trend = 1
                        sar[i] = ep
                        ep = high.iloc[i]
                        af = acceleration
                    else:
                        if low.iloc[i] < ep:
                            ep = low.iloc[i]
                            af = min(af + acceleration, maximum)
        return pd.Series(sar, index=close.index)
    except Exception as e:
        print(f"Error in compute_parabolic_sar: {e}")

def compute_ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_span_b_period=52):
    try:
        tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
        kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        senkou_span_b = ((high.rolling(window=senkou_span_b_period).max() + low.rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)
        chikou_span = close.shift(-kijun_period)
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    except Exception as e:
        print(f"Error in compute_ichimoku: {e}")

def compute_fibonacci_retracement(high, low):
    try:
        levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        retracement_levels = {}
        for level in levels:
            retracement_levels[level] = high - (high - low) * level
        return retracement_levels
    except Exception as e:
        print(f"Error in compute_fibonacci_retracement: {e}")

def compute_volume_profile(close, volume, bins=20):
    try:
        hist, bin_edges = np.histogram(close, bins=bins, weights=volume)
        volume_profile = np.full(len(close), np.nan)
        volume_profile[:len(hist)] = hist
        return volume_profile, bin_edges
    except Exception as e:
        print(f"Error in compute_volume_profile: {e}")

def detect_candlestick_patterns(df):
    try:
        patterns = []
        for i in range(len(df)):
            if i == 0:
                patterns.append(None)
            else:
                if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                    patterns.append("Bullish Engulfing")
                elif df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                    patterns.append("Bearish Engulfing")
                else:
                    patterns.append(None)
        return patterns
    except Exception as e:
        print(f"Error in detect_candlestick_patterns: {e}")

def compute_trendlines(high, low):
    try:
        x = np.arange(len(high))
        slope, intercept, _, _, _ = linregress(x, high)
        trendline_high = slope * x + intercept
        slope, intercept, _, _, _ = linregress(x, low)
        trendline_low = slope * x + intercept
        return trendline_high, trendline_low
    except Exception as e:
        print(f"Error in compute_trendlines: {e}")

def compute_volatility(prices: Union[np.ndarray, pd.Series, float], period: int = 14) -> float:
    """
    Compute volatility using standard deviation of returns
    
    :param prices: Price data (can be a single price or a series)
    :param period: Lookback period for volatility calculation
    :return: Volatility measure
    """
    try:
        # If input is a single price, convert to numpy array
        if isinstance(prices, (int, float, np.float64)):
            return 0.0  # Cannot compute volatility for a single price
        
        # Ensure we have a pandas Series or numpy array
        if not isinstance(prices, (pd.Series, np.ndarray)):
            prices = pd.Series(prices)
        
        # Compute returns
        returns = prices.pct_change().dropna()
        
        # Compute volatility (standard deviation of returns)
        volatility = returns.std() * np.sqrt(period)
        
        return float(volatility) if not np.isnan(volatility) else 0.0
    
    except Exception as e:
        logger.error(f"Error in compute_volatility: {e}")
        return 0.0

def fetch_market_data(symbol: str) -> Dict:
    """
    Fetch market data for a specific trading symbol from MetaTrader5
    
    :param symbol: Trading symbol to fetch data for
    :return: Dictionary containing market data
    """
    try:
        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Could not select symbol {symbol}")
            return {}
        
        # Get current tick information
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"No tick data available for {symbol}")
            return {}
        
        # Fetch recent rates
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 100)
        if rates is None or len(rates) == 0:
            logger.warning(f"No historical rates available for {symbol}")
            return {}
        
        # Convert rates to DataFrame for easier processing
        df = pd.DataFrame(rates)
        
        # Prepare market data dictionary
        market_data = {
            'symbol': symbol,
            'open': df['open'].iloc[-1],
            'high': df['high'].iloc[-1],
            'low': df['low'].iloc[-1],
            'close': df['close'].iloc[-1],
            'tick_volume': df['tick_volume'].iloc[-1],
            'spread': mt5.symbol_info(symbol).spread,
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': pd.to_datetime(df['time'].iloc[-1], unit='s')
        }
        
        logger.debug(f"Market data fetched for {symbol}: {market_data}")
        return market_data
    
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return {}

def analyze_market_conditions(market_data: Dict) -> Dict:
    """
    Analyze market conditions for a specific symbol
    
    :param market_data: Market data dictionary
    :return: Market conditions dictionary
    """
    try:
        # Compute volatility
        volatility = compute_volatility(market_data['close'])
        
        # Detect trend changes
        trend_change = detect_trend_change(market_data)
        
        market_conditions = {
            'volatility': volatility,
            'trend_change': trend_change,
            'volatility_spike': volatility > 1.5,  # Example threshold
        }
        
        return market_conditions
    
    except Exception as e:
        logger.error(f"Error analyzing market conditions: {e}")
        return {
            'volatility': 0.0,
            'trend_change': False,
            'volatility_spike': False
        }

def generate_trading_signal(market_data: Dict) -> Dict:
    """
    Generate trading signals based on multiple technical indicators with adaptive criteria
    
    :param market_data: Market data for a single symbol
    :return: Trading signal dictionary with detailed reasoning
    """
    try:
        # Detect trend change with multi-indicator confirmation
        trend_change = detect_trend_change(market_data)
        
        symbol = market_data.get('symbol')
        current_price = market_data.get('close')
        
        if not symbol or current_price is None:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient market data'
            }
        
        # Fetch historical data for comprehensive analysis
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 300)
        
        if rates is None or len(rates) == 0:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': 'No historical data available'
            }
        
        df = pd.DataFrame(rates)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Compute comprehensive indicators
        rsi = compute_rsi(df['close'])
        macd_line, signal_line, histogram = compute_macd(df['close'])
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df['close'])
        atr = compute_atr(df['high'], df['low'], df['close'])
        
        # Adaptive Signal Generation
        reasons = []
        signal_score = 0
        
        # RSI Analysis
        if rsi.iloc[-1] < 30:
            reasons.append("Oversold Condition")
            signal_score += 0.3
        elif rsi.iloc[-1] > 70:
            reasons.append("Overbought Condition")
            signal_score -= 0.3
        
        # MACD Analysis
        if macd_line[-1] > signal_line[-1] and histogram[-1] > 0:
            reasons.append("Bullish MACD Momentum")
            signal_score += 0.4
        elif macd_line[-1] < signal_line[-1] and histogram[-1] < 0:
            reasons.append("Bearish MACD Momentum")
            signal_score -= 0.4
        
        # Bollinger Bands Analysis
        if current_price < bb_lower[-1]:
            reasons.append("Price Near Lower Bollinger Band")
            signal_score += 0.2
        elif current_price > bb_upper[-1]:
            reasons.append("Price Near Upper Bollinger Band")
            signal_score -= 0.2
        
        # ATR Volatility Check
        if len(atr) > 0 and atr[-1] > 0:
            volatility = atr[-1]
            reasons.append(f"Current Volatility: {volatility:.4f}")
        
        # Determine Signal
        if signal_score > 0.5:
            return {
                'signal': 'BUY',
                'confidence': min(signal_score, 1.0),
                'entry_price': current_price,
                'stop_loss': bb_lower[-1],
                'take_profit': current_price * 1.02,
                'reasons': reasons
            }
        elif signal_score < -0.5:
            return {
                'signal': 'SELL',
                'confidence': min(abs(signal_score), 1.0),
                'entry_price': current_price,
                'stop_loss': bb_upper[-1],
                'take_profit': current_price * 0.98,
                'reasons': reasons
            }
        
        # Fallback to HOLD with detailed reasoning
        return {
            'signal': 'HOLD',
            'confidence': abs(signal_score),
            'reasons': reasons or ['No Clear Trading Signal']
        }
    
    except Exception as e:
        logger.error(f"Error generating trading signal: {e}")
        return {
            'signal': 'HOLD',
            'confidence': 0.0,
            'reason': str(e)
        }

def detect_trend_change(market_data: Dict) -> Dict[str, Union[bool, float]]:
    """
    Detect potential trend changes in market data with multi-indicator confirmation
    
    :param market_data: Market data dictionary
    :return: Dictionary with trend change status and confidence score
    """
    try:
        symbol = market_data.get('symbol')
        if not symbol:
            raise ValueError("Symbol not provided in market data")
        
        # Fetch more historical data for trend analysis
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 300)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No historical data available for trend analysis of {symbol}")
            return {"trend_change": False, "confidence": 0.0}
        
        # Convert rates to DataFrame
        df = pd.DataFrame(rates)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
        # Compute multiple indicators with different windows for more nuanced analysis
        ma_fast = df['close'].rolling(window=10).mean()
        ma_medium = df['close'].rolling(window=20).mean()
        ma_slow = df['close'].rolling(window=50).mean()
        
        rsi = compute_rsi(df['close'])
        macd_line, signal_line, histogram = compute_macd(df['close'])
        
        # Advanced Moving Average Crossover Detection
        ma_crossover = (
            # Fast MA crossing Medium MA
            (ma_fast.iloc[-2] <= ma_medium.iloc[-2]) and 
            (ma_fast.iloc[-1] > ma_medium.iloc[-1])
        ) or (
            (ma_fast.iloc[-2] >= ma_medium.iloc[-2]) and 
            (ma_fast.iloc[-1] < ma_medium.iloc[-1])
        ) or (
            # Medium MA crossing Slow MA
            (ma_medium.iloc[-2] <= ma_slow.iloc[-2]) and 
            (ma_medium.iloc[-1] > ma_slow.iloc[-1])
        ) or (
            (ma_medium.iloc[-2] >= ma_slow.iloc[-2]) and 
            (ma_medium.iloc[-1] < ma_slow.iloc[-1])
        )
        
        # RSI Confirmation with Expanded Range
        rsi_confirmation = (
            (rsi.iloc[-1] > 20 and rsi.iloc[-1] < 80) and  # Wider range
            (abs(rsi.iloc[-1] - rsi.iloc[-2]) > 5)  # Significant RSI change
        )
        
        # MACD Confirmation with Histogram Strength
        macd_confirmation = (
            (macd_line[-1] > signal_line[-1]) or  # Bullish
            (macd_line[-1] < signal_line[-1])     # Bearish
        ) and (
            abs(histogram[-1]) > abs(histogram[-2])  # Increasing momentum
        )
        
        # Compute confidence score with more granular weighting
        confidence = 0.0
        if ma_crossover:
            confidence += 0.4
        if rsi_confirmation:
            confidence += 0.3
        if macd_confirmation:
            confidence += 0.3
        
        # Trend change requires multiple confirmations
        trend_change = ma_crossover and rsi_confirmation and macd_confirmation
        
        # Log detailed trend analysis
        logger.debug(f"Trend Change Analysis for {symbol}: "
                     f"MA Crossover={ma_crossover}, "
                     f"RSI Confirmation={rsi_confirmation}, "
                     f"MACD Confirmation={macd_confirmation}, "
                     f"Confidence={confidence}")
        
        return {
            "trend_change": trend_change, 
            "confidence": confidence
        }
    
    except Exception as e:
        logger.error(f"Error detecting trend change: {e}")
        return {"trend_change": False, "confidence": 0.0}

async def analyze_candles(df):
    try:
        df['MA'] = df['close'].rolling(window=14).mean()
        df['EMA'] = df['close'].ewm(span=14, adjust=False).mean()
        df['RSI'] = compute_rsi(df['close'], 14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df['close'])
        df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['close'], 20, 2)
        df['ADX'] = compute_adx(df['high'], df['low'], df['close'], 14)
        df['Momentum'] = compute_momentum(df['close'], 14)
        df['Stochastic_K'], df['Stochastic_D'] = compute_stochastic(df['high'], df['low'], df['close'])
        df['ATR'] = compute_atr(df['high'], df['low'], df['close'])
        df['OBV'] = compute_obv(df['close'], df['tick_volume'])
        df['CCI'] = compute_cci(df['high'], df['low'], df['close'])
        df['MFI'] = compute_mfi(df['high'], df['low'], df['close'], df['tick_volume'])
        df['Williams_R'] = compute_williams_r(df['high'], df['low'], df['close'])
        df['VWAP'] = compute_vwap(df['high'], df['low'], df['close'], df['tick_volume'])
        df['Supertrend'] = compute_supertrend(df['high'], df['low'], df['close'])
        df['Parabolic_SAR'] = compute_parabolic_sar(df['high'], df['low'], df['close'])
        df['Tenkan_Sen'], df['Kijun_Sen'], df['Senkou_Span_A'], df['Senkou_Span_B'], df['Chikou_Span'] = compute_ichimoku(df['high'], df['low'], df['close'])
        df['Fibonacci_Retracement'] = compute_fibonacci_retracement(df['high'], df['low'])
        df['Volume_Profile'], _ = compute_volume_profile(df['close'], df['tick_volume'])
        df['Candlestick_Patterns'] = detect_candlestick_patterns(df)
        df['Trendline_High'], df['Trendline_Low'] = compute_trendlines(df['high'], df['low'])
        df['Volatility'] = compute_volatility(df['close'])
        df['Position_Size'] = calculate_position_size(10000, 1, 10)  # مثال: حساب حجم المركز
        df['Sentiment'] = analyze_sentiment("Example text")  # مثال: تحليل المشاعر
        df['ML_Prediction'] = predict_with_ml(df)  # مثال: التنبؤ باستخدام التعلم الآلي
        
        # Handle NaN values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        filled_data = imputer.fit_transform(df[['MA', 'RSI', 'MACD']])

        # Data Normalization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(filled_data)

        # Feature Selection using SelectKBest
        from sklearn.feature_selection import SelectKBest
        selector = SelectKBest(score_func=f_regression, k='all')
        selected_features = selector.fit_transform(normalized_data, np.random.rand(100))

        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(selected_features, np.random.rand(100), test_size=0.2, random_state=42)

        # إعداد شبكة LSTM مع معاملات قابلة للتعديل
        def create_lstm_model(input_shape, units=64, dropout=0.2, optimizer='adam'):
            model = Sequential([
                LSTM(units=units, input_shape=input_shape, return_sequences=True),
                Dropout(dropout),
                LSTM(units=units//2),
                Dropout(dropout),
                Dense(1)
            ])
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model

        # دالة للبحث عن أفضل المعاملات
        def find_best_hyperparameters(X_train, y_train, X_test, y_test):
            # معاملات للبحث
            units_list = [32, 64, 128]
            dropout_rates = [0.2, 0.3, 0.4]
            optimizers = ['adam', 'rmsprop']

            best_score = float('inf')
            best_params = {}
            best_model = None

            # البحث عن أفضل المعاملات
            for units in units_list:
                for dropout in dropout_rates:
                    for optimizer in optimizers:
                        try:
                            # إعادة تعيين البذرة العشوائية
                            tf.random.set_seed(42)
                            np.random.seed(42)

                            # إنشاء النموذج
                            model = create_lstm_model(
                                input_shape=(X_train.shape[1], 1), 
                                units=units, 
                                dropout=dropout, 
                                optimizer=optimizer
                            )

                            # التدريب
                            from tensorflow.keras.callbacks import EarlyStopping
                            early_stopping = EarlyStopping(
                                monitor='val_loss', 
                                patience=10, 
                                restore_best_weights=True
                            )
                            
                            history = model.fit(
                                X_train, y_train, 
                                validation_split=0.2,
                                epochs=100, 
                                batch_size=32, 
                                verbose=0,
                                callbacks=[early_stopping]
                            )

                            # التقييم
                            predictions = model.predict(X_test).flatten()
                            mse = mean_squared_error(y_test, predictions)

                            # تحديث أفضل النتائج
                            if mse < best_score:
                                best_score = mse
                                best_params = {
                                    'units': units,
                                    'dropout': dropout,
                                    'optimizer': optimizer
                                }
                                best_model = model

                        except Exception as inner_e:
                            logger.error(f"خطأ في التدريب للتكوين: {units}, {dropout}, {optimizer}")
                            logger.error(traceback.format_exc())

            return best_model, best_params, best_score

        # إعادة تشكيل البيانات للـ LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # العثور على أفضل المعاملات
        best_model, best_params, best_score = find_best_hyperparameters(
            X_train, y_train, X_test, y_test
        )

        # التنبؤ والتقييم
        predictions = best_model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # عرض النتائج
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="نتائج تقييم النموذج")
        table.add_column("المقياس", justify="right", style="cyan", no_wrap=True)
        table.add_column("القيمة", style="magenta")
        table.add_row("أفضل المعاملات", str(best_params))
        table.add_row("متوسط مربع الخطأ (MSE)", f"{mse:.4f}")
        table.add_row("الجذر التربيعي لمتوسط مربع الخطأ (RMSE)", f"{rmse:.4f}")
        table.add_row("متوسط الخطأ المطلق (MAE)", f"{mae:.4f}")
        table.add_row("معامل التحديد (R²)", f"{r2:.4f}")
        table.add_row("متوسط النسبة المئوية للخطأ المطلق (MAPE)", f"{mape:.2f}%")

        console.print(table)

        # Dimensionality Reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(normalized_data)

        # Anomaly Detection
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1)
        model.fit(reduced_data)
        import joblib
        joblib.dump(model, 'isolation_forest_model.pkl')

        loaded_model = joblib.load('isolation_forest_model.pkl')
        anomalies = loaded_model.predict(reduced_data)

        # Sentiment Analysis Example
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores("This is a great day!")

        # Automated Reporting (simple visualization)
        import matplotlib.pyplot as plt
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=anomalies)
        plt.title('Anomaly Detection')
        # plt.show()  # Commented out to prevent the plot window from appearing

        return df
    except Exception as e:
        print(f"Error in analyze_candles: {e}")

async def main():
    try:
        # Check if the CSV file exists, if not create it with random data
        csv_file_path = 'real_data.csv'
        if not os.path.exists(csv_file_path):
            # Create a DataFrame with random data
            data = pd.DataFrame({
                'feature1': np.random.rand(100),
                'feature2': np.random.rand(100)
            })
            # Save the DataFrame to a CSV file
            data.to_csv(csv_file_path, index=False)
        else:
            # Load the existing CSV file
            data = pd.read_csv(csv_file_path)

        # Handle NaN values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        filled_data = imputer.fit_transform(data)

        # Data Normalization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(filled_data)

        # Feature Selection using SelectKBest
        from sklearn.feature_selection import SelectKBest
        selector = SelectKBest(score_func=f_regression, k='all')
        selected_features = selector.fit_transform(normalized_data, np.random.rand(100))

        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(selected_features, np.random.rand(100), test_size=0.2, random_state=42)

        # إعداد شبكة LSTM مع معاملات قابلة للتعديل
        def create_lstm_model(input_shape, units=64, dropout=0.2, optimizer='adam'):
            model = Sequential([
                LSTM(units=units, input_shape=input_shape, return_sequences=True),
                Dropout(dropout),
                LSTM(units=units//2),
                Dropout(dropout),
                Dense(1)
            ])
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model

        # دالة للبحث عن أفضل المعاملات
        def find_best_hyperparameters(X_train, y_train, X_test, y_test):
            # معاملات للبحث
            units_list = [32, 64, 128]
            dropout_rates = [0.2, 0.3, 0.4]
            optimizers = ['adam', 'rmsprop']

            best_score = float('inf')
            best_params = {}
            best_model = None

            # البحث عن أفضل المعاملات
            for units in units_list:
                for dropout in dropout_rates:
                    for optimizer in optimizers:
                        try:
                            # إعادة تعيين البذرة العشوائية
                            tf.random.set_seed(42)
                            np.random.seed(42)

                            # إنشاء النموذج
                            model = create_lstm_model(
                                input_shape=(X_train.shape[1], 1), 
                                units=units, 
                                dropout=dropout, 
                                optimizer=optimizer
                            )

                            # التدريب
                            from tensorflow.keras.callbacks import EarlyStopping
                            early_stopping = EarlyStopping(
                                monitor='val_loss', 
                                patience=10, 
                                restore_best_weights=True
                            )
                            
                            history = model.fit(
                                X_train, y_train, 
                                validation_split=0.2,
                                epochs=100, 
                                batch_size=32, 
                                verbose=0,
                                callbacks=[early_stopping]
                            )

                            # التقييم
                            predictions = model.predict(X_test).flatten()
                            mse = mean_squared_error(y_test, predictions)

                            # تحديث أفضل النتائج
                            if mse < best_score:
                                best_score = mse
                                best_params = {
                                    'units': units,
                                    'dropout': dropout,
                                    'optimizer': optimizer
                                }
                                best_model = model

                        except Exception as inner_e:
                            logger.error(f"خطأ في التدريب للتكوين: {units}, {dropout}, {optimizer}")
                            logger.error(traceback.format_exc())

            return best_model, best_params, best_score

        # إعادة تشكيل البيانات للـ LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # العثور على أفضل المعاملات
        best_model, best_params, best_score = find_best_hyperparameters(
            X_train, y_train, X_test, y_test
        )

        # التنبؤ والتقييم
        predictions = best_model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # عرض النتائج
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="نتائج تقييم النموذج")
        table.add_column("المقياس", justify="right", style="cyan", no_wrap=True)
        table.add_column("القيمة", style="magenta")
        table.add_row("أفضل المعاملات", str(best_params))
        table.add_row("متوسط مربع الخطأ (MSE)", f"{mse:.4f}")
        table.add_row("الجذر التربيعي لمتوسط مربع الخطأ (RMSE)", f"{rmse:.4f}")
        table.add_row("متوسط الخطأ المطلق (MAE)", f"{mae:.4f}")
        table.add_row("معامل التحديد (R²)", f"{r2:.4f}")
        table.add_row("متوسط النسبة المئوية للخطأ المطلق (MAPE)", f"{mape:.2f}%")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error in main loop:[/bold red] {e}")
        logging.error("Exception occurred", exc_info=True)

def main():
    try:
        # Configuration
        initial_balance = 10000  # USD
        trading_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']  # Multi-symbol support
        
        # Start multi-symbol trading
        multi_symbol_trading(trading_symbols, initial_balance)
    
    except Exception as e:
        logger.error(f"Critical error in main trading loop: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # Ensure MT5 connection is properly closed
        if MT5_INITIALIZED:
            mt5.shutdown()

class TradeManager:
    def __init__(self, symbols: List[str], initial_balance: float, risk_percentage: float = 1.0):
        """
        Initialize trade manager with multi-symbol support and dynamic risk management
        
        :param symbols: List of trading symbols
        :param initial_balance: Total account balance
        :param risk_percentage: Base risk percentage
        """
        self.symbols = symbols
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.base_risk_percentage = risk_percentage
        self.active_trades: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
    def compute_dynamic_risk(self, symbol: str, market_volatility: float) -> float:
        """
        Dynamically adjust risk based on market conditions
        
        :param symbol: Trading symbol
        :param market_volatility: Current market volatility
        :return: Adjusted risk percentage
        """
        base_risk = self.base_risk_percentage
        volatility_factor = min(max(market_volatility, 0.5), 2.0)
        return base_risk * volatility_factor
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size with advanced risk management
        
        :param symbol: Trading symbol
        :param entry_price: Entry price
        :param stop_loss: Stop loss price
        :return: Position size in lots
        """
        risk_amount = self.current_balance * self.base_risk_percentage / 100
        risk_per_share = abs(entry_price - stop_loss)
        position_size = risk_amount / risk_per_share
        
        return min(position_size, self.current_balance / entry_price)
    
    def open_trade(self, symbol: str, trade_type: str, entry_price: float, stop_loss: float, take_profit: float):
        """
        Open a trade with comprehensive tracking
        
        :param symbol: Trading symbol
        :param trade_type: 'BUY' or 'SELL'
        :param entry_price: Entry price
        :param stop_loss: Stop loss price
        :param take_profit: Take profit price
        """
        try:
            position_size = self.calculate_position_size(symbol, entry_price, stop_loss)
            
            trade_details = {
                'symbol': symbol,
                'type': trade_type,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'open_time': datetime.now(),
                'status': 'OPEN'
            }
            
            self.active_trades[symbol] = trade_details
            logger.info(f"Trade opened: {trade_details}")
        except Exception as e:
            logger.error(f"Error opening trade for {symbol}: {e}")
    
    def manage_open_trades(self, current_prices: Dict[str, float], market_conditions: Dict[str, Dict]):
        """
        Dynamically manage open trades
        
        :param current_prices: Current prices for each symbol
        :param market_conditions: Market condition details for each symbol
        """
        for symbol, trade in list(self.active_trades.items()):
            current_price = current_prices.get(symbol)
            conditions = market_conditions.get(symbol, {})
            
            if not current_price:
                continue
            
            # Dynamic stop loss and take profit adjustment
            if conditions.get('trend_change') or conditions.get('volatility_spike'):
                self.adjust_trade_parameters(symbol, current_price, conditions)
            
            # Check for trade closure conditions
            if self.should_close_trade(trade, current_price):
                self.close_trade(symbol)
    
    def adjust_trade_parameters(self, symbol: str, current_price: float, conditions: Dict):
        """
        Dynamically adjust trade parameters
        """
        trade = self.active_trades.get(symbol)
        if not trade:
            return
        
        # Example adjustments based on market conditions
        if conditions.get('trend_change'):
            trade['stop_loss'] = current_price * (0.95 if trade['type'] == 'BUY' else 1.05)
        
        if conditions.get('volatility_spike'):
            trade['take_profit'] = current_price * (1.02 if trade['type'] == 'BUY' else 0.98)
    
    def should_close_trade(self, trade: Dict, current_price: float) -> bool:
        """
        Determine if a trade should be closed
        """
        if trade['type'] == 'BUY':
            return (current_price <= trade['stop_loss']) or (current_price >= trade['take_profit'])
        else:  # SELL
            return (current_price >= trade['stop_loss']) or (current_price <= trade['take_profit'])
    
    def close_trade(self, symbol: str):
        """
        Close a trade and update trade history
        """
        trade = self.active_trades.pop(symbol, None)
        if trade:
            trade['close_time'] = datetime.now()
            trade['status'] = 'CLOSED'
            self.trade_history.append(trade)
            logger.info(f"Trade closed: {trade}")
    
    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive trading performance report
        """
        total_trades = len(self.trade_history)
        profitable_trades = sum(1 for trade in self.trade_history if 
                                (trade['type'] == 'BUY' and trade['close_price'] > trade['entry_price']) or
                                (trade['type'] == 'SELL' and trade['close_price'] < trade['entry_price']))
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': profitable_trades / total_trades if total_trades > 0 else 0,
            'total_profit': self.current_balance - self.initial_balance,
            'profit_percentage': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        }

def multi_symbol_trading(symbols: List[str], initial_balance: float, max_iterations: int = 10):
    """
    Perform trading across multiple symbols with iteration limit and comprehensive logging
    
    :param symbols: List of trading symbols
    :param initial_balance: Initial trading account balance
    :param max_iterations: Maximum number of trading iterations
    """
    open_trades = {}
    iteration = 0
    
    while iteration < max_iterations:
        try:
            logger.info(f"Trading Iteration {iteration + 1}")
            
            # Check for reporting time and generate performance report
            if is_reporting_time():
                performance_report = calculate_performance(open_trades)
                save_performance_report(performance_report)
            
            for symbol in symbols:
                try:
                    # Fetch market data for individual symbol
                    market_data = fetch_market_data(symbol)
                    
                    # Skip if no market data
                    if not market_data:
                        logger.warning(f"No market data for {symbol}. Skipping.")
                        continue
                    
                    # Analyze market conditions
                    market_conditions = analyze_market_conditions(market_data)
                    
                    # Generate trading signal
                    signal = generate_trading_signal(market_data)
                    logger.info(f"Signal for {symbol}: {signal}")
                    
                    # Execute trade if signal is not HOLD
                    if signal['signal'] != 'HOLD':
                        trade_result = execute_trade(signal, symbol, initial_balance)
                        
                        # Track open trades
                        if trade_result.get('status') == 'OPEN':
                            open_trades[symbol] = trade_result
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Wait before next iteration
            time.sleep(TRADING_INTERVAL)
            iteration += 1
        
        except Exception as e:
            logger.error(f"Critical error in multi-symbol trading: {e}")
            time.sleep(ERROR_RECOVERY_INTERVAL)
    
    # Close remaining open trades at end of iterations
    for symbol, trade in open_trades.items():
        try:
            close_trade(trade)
        except Exception as e:
            logger.error(f"Error closing trade for {symbol}: {e}")
    
    logger.info("Multi-symbol trading completed")

def execute_trade(signal: Dict, symbol: str, initial_balance: float) -> Dict:
    """
    Execute a trade in MetaTrader5 based on the generated signal
    
    :param signal: Trading signal dictionary
    :param symbol: Trading symbol
    :param initial_balance: Initial account balance
    :return: Trade execution result dictionary
    """
    try:
        # Validate signal
        if signal['signal'] in ['HOLD', None]:
            logger.info(f"No trade execution for {symbol}: {signal.get('reason', 'No signal')}")
            return {'status': 'NO_TRADE'}
        
        # Fetch current market information
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found in Market Watch")
            return {'status': 'ERROR', 'reason': 'Symbol not found'}
        
        # Ensure symbol is selected in Market Watch
        if not symbol_info.visible:
            logger.info(f"Selecting {symbol} in Market Watch")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Could not select {symbol}")
                return {'status': 'ERROR', 'reason': 'Cannot select symbol'}
        
        # Get current market price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Could not get tick data for {symbol}")
            return {'status': 'ERROR', 'reason': 'No tick data'}
        
        # Determine trade type and price
        if signal['signal'] == 'BUY':
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            stop_loss = signal.get('stop_loss', price * 0.99)
            take_profit = signal.get('take_profit', price * 1.01)
        elif signal['signal'] == 'SELL':
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            stop_loss = signal.get('stop_loss', price * 1.01)
            take_profit = signal.get('take_profit', price * 0.99)
        else:
            logger.error(f"Invalid trade signal: {signal['signal']}")
            return {'status': 'ERROR', 'reason': 'Invalid signal'}
        
        # Calculate position size (risk management)
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Could not retrieve account information")
            return {'status': 'ERROR', 'reason': 'Account info unavailable'}
        
        # Risk management: 2% of account balance per trade
        risk_percentage = 0.02
        account_balance = account_info.balance
        risk_amount = account_balance * risk_percentage
        
        # Calculate stop loss distance
        stop_loss_distance = abs(price - stop_loss)
        
        # Calculate position size
        tick_value = mt5.symbol_info(symbol).trade_tick_value
        lot_size = risk_amount / (stop_loss_distance * tick_value)
        lot_size = round(lot_size, 2)  # Round to 2 decimal places
        
        # Ensure lot size is within broker's limits
        lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
        
        # Prepare trade request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": trade_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,  # Allow 10 points of slippage
            "magic": 234000,  # Unique identifier for the strategy
            "comment": f"{signal['signal']} {','.join(signal.get('reasons', []))}",
            "type_time": mt5.ORDER_TIME_GTC  # Good Till Canceled
        }
        
        # Send trade request
        result = mt5.order_send(request)
        
        # Check trade execution result
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            trade_info = {
                'symbol': symbol,
                'type': signal['signal'],
                'entry_price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': lot_size,
                'open_time': datetime.now(),
                'status': 'OPEN',
                'reasons': signal.get('reasons', [])
            }
            logger.info(f"Trade opened: {trade_info}")
            return trade_info
        else:
            logger.error(f"Trade execution failed for {symbol}. Error: {result.comment}")
            return {
                'status': 'ERROR',
                'reason': result.comment,
                'retcode': result.retcode
            }
    
    except Exception as e:
        logger.error(f"Critical error in trade execution for {symbol}: {e}")
        return {'status': 'ERROR', 'reason': str(e)}

# Trading configuration constants
TRADING_INTERVAL = 60  # Trading check interval in seconds
ERROR_RECOVERY_INTERVAL = 30  # Error recovery wait time in seconds
MAX_TRADE_RETRIES = 3  # Maximum number of trade retry attempts

def is_reporting_time() -> bool:
    """
    Determine if it's time to generate a performance report
    
    Default implementation: Report daily at midnight
    Can be customized based on specific reporting needs
    """
    current_time = datetime.now()
    return current_time.hour == 0 and current_time.minute == 0

def save_performance_report(report: Dict):
    """
    Save performance report to a file
    
    :param report: Performance report dictionary
    """
    try:
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Performance report saved: {filename}")
    except Exception as e:
        logger.error(f"Error saving performance report: {e}")

def calculate_performance(open_trades: Dict) -> Dict:
    """
    Calculate trading performance based on open trades
    
    :param open_trades: Dictionary of currently open trades
    :return: Performance report dictionary
    """
    try:
        account_info = mt5.account_info()
        
        performance_report = {
            'timestamp': datetime.now().isoformat(),
            'account_balance': account_info.balance if account_info else 0,
            'open_trades_count': len(open_trades),
            'open_trades': {},
            'total_profit': 0.0
        }
        
        # Collect details for each open trade
        for symbol, trade in open_trades.items():
            performance_report['open_trades'][symbol] = {
                'entry_price': trade.get('entry_price', 0),
                'current_price': mt5.symbol_info_tick(symbol).last if mt5.symbol_info_tick(symbol) else 0,
                'type': trade.get('type', 'UNKNOWN'),
                'reasons': trade.get('reasons', [])
            }
        
        return performance_report
    
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def close_trade(trade: Dict):
    """
    Close an open trade
    
    :param trade: Trade dictionary to close
    """
    try:
        symbol = trade.get('symbol')
        if not symbol:
            logger.warning("Cannot close trade: No symbol specified")
            return
        
        # Prepare trade closure request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": trade.get('position_size', 0),
            "type": mt5.ORDER_TYPE_SELL if trade.get('type') == 'BUY' else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).bid if trade.get('type') == 'BUY' else mt5.symbol_info_tick(symbol).ask,
            "deviation": 10,
            "magic": 234000,
            "comment": "Trade closure"
        }
        
        # Send trade closure request
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Successfully closed trade for {symbol}")
        else:
            logger.error(f"Failed to close trade for {symbol}. Error: {result.comment}")
    
    except Exception as e:
        logger.error(f"Error closing trade: {e}")

if __name__ == "__main__":
    main()