# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf

from .base import BaseFetcher, DataFetchError
from .realtime_types import UnifiedRealtimeQuote
from src.analyzer import STOCK_NAME_MAP

logger = logging.getLogger(__name__)

class YfinanceFetcher(BaseFetcher):
    name = "YfinanceFetcher"
    priority = 0

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            # 往前推 60 天以確保 MA20/MA10 能夠完美計算
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)
            start_date_yf = start_dt.strftime("%Y-%m-%d")
            
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            end_date_yf = end_dt.strftime("%Y-%m-%d")
            
            logger.info(f"Yfinance 请求历史数据: {stock_code}")
            ticker = yf.Ticker(stock_code)
            df = ticker.history(start=start_date_yf, end=end_date_yf)
            
            if df.empty:
                raise DataFetchError(f"Yahoo Finance 返回空数据: {stock_code}")
                
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            raise DataFetchError(f"Yahoo Finance 获取历史数据失败: {e}")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()
        date_col = 'Date' if 'Date' in df.columns else 'Datetime'
        df['date'] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        
        df['open'] = df['Open']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['close'] = df['Close']
        df['volume'] = df['Volume']
        df['amount'] = df['volume'] * df['close']
        
        df['pct_chg'] = df['close'].pct_change() * 100
        df['pct_chg'] = df['pct_chg'].fillna(0.0)
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']]

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算均線與量比，修復 0 或 N/A 的問題"""
        df = df.copy()
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        try:
            ticker = yf.Ticker(stock_code)
            df = ticker.history(period="5d")
            if df.empty:
                return None
                
            current_row = df.iloc[-1]
            current_price = float(current_row['Close'])
            open_price = float(current_row['Open'])
            high_price = float(current_row['High'])
            low_price = float(current_row['Low'])
            volume = float(current_row['Volume'])
            
            previous_close = current_price
            if len(df) > 1:
                previous_close = float(df.iloc[-2]['Close'])
                
            change = current_price - previous_close
            change_pct = (change / previous_close * 100) if previous_close else 0.0
            amplitude = ((high_price - low_price) / previous_close * 100) if previous_close else 0.0
            
            name = STOCK_NAME_MAP.get(stock_code, stock_code)

            quote = UnifiedRealtimeQuote(
                stock_code=stock_code,
                name=name,
                current_price=current_price,
                change=change,
                change_pct=change_pct,
                open_price=open_price,   # 補上開盤價
                high_price=high_price,   # 補上最高價
                low_price=low_price,     # 補上最低價
                volume=volume,
                amount=volume * current_price,
                turnover_rate=None, 
                amplitude=amplitude,
                volume_ratio=None,
                pe_ratio=None,
                pb_ratio=None,
                total_mv=None,
                circ_mv=None,
                source="yfinance"
            )
            return quote
        except Exception as e:
            return None

    def get_stock_name(self, stock_code: str) -> str:
        return STOCK_NAME_MAP.get(stock_code, stock_code)
