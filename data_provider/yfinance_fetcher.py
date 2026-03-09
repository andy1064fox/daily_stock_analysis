# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf

from .base import BaseFetcher, DataFetchError
from .realtime_types import UnifiedRealtimeQuote

logger = logging.getLogger(__name__)

class YfinanceFetcher(BaseFetcher):
    name = "YfinanceFetcher"
    priority = 0

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            # yfinance 的結束日期是不包含的，所以要往後加 1 天
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            end_date_yf = end_dt.strftime("%Y-%m-%d")
            
            ticker = yf.Ticker(stock_code)
            df = ticker.history(start=start_date, end=end_date_yf)
            
            if df.empty:
                raise DataFetchError(f"Yahoo Finance 返回空数据: {stock_code}")
                
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            raise DataFetchError(f"Yahoo Finance 获取历史数据失败: {e}")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()
        
        # 處理時區問題，並格式化日期
        df['date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        
        # 映射標準欄位
        df['open'] = df['Open']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['close'] = df['Close']
        df['volume'] = df['Volume']
        
        # 近似計算成交額 (成交量 * 收盤價)
        df['amount'] = df['volume'] * df['close']
        
        # 計算漲跌幅 (百分比)
        df['pct_chg'] = df['close'].pct_change() * 100
        df['pct_chg'] = df['pct_chg'].fillna(0.0)
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']]

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        try:
            ticker = yf.Ticker(stock_code)
            # 使用 fast_info 獲取即時資料，速度快且不易被限流
            fast_info = ticker.fast_info
            info = ticker.info
            
            current_price = fast_info.get('lastPrice', info.get('currentPrice'))
            previous_close = fast_info.get('previousClose', info.get('previousClose'))
            
            if current_price is None or previous_close is None:
                return None
                
            open_price = fast_info.get('open', current_price)
            high_price = fast_info.get('dayHigh', current_price)
            low_price = fast_info.get('dayLow', current_price)
            volume = fast_info.get('lastVolume', 0)
            
            # 計算漲跌數值
            change = current_price - previous_close
            change_pct = (change / previous_close * 100) if previous_close else 0.0
            amplitude = ((high_price - low_price) / previous_close * 100) if previous_close else 0.0
            
            # 獲取股票名稱 (例如：Taiwan Semiconductor)
            name = info.get('shortName') or info.get('longName') or stock_code

            quote = UnifiedRealtimeQuote(
                stock_code=stock_code,
                name=name,
                current_price=float(current_price),
                change=float(change),
                change_pct=float(change_pct),
                open_price=float(open_price),
                high_price=float(high_price),
                low_price=float(low_price),
                volume=float(volume),
                amount=float(volume * current_price),
                turnover_rate=None, 
                amplitude=float(amplitude),
                volume_ratio=None,
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                total_mv=info.get('marketCap'),
                circ_mv=info.get('marketCap'),
                source="yfinance"
            )
            return quote
        except Exception as e:
            logger.warning(f"Yahoo Finance 获取 {stock_code} 实时行情失败: {e}")
            return None

    def get_stock_name(self, stock_code: str) -> str:
        try:
            info = yf.Ticker(stock_code).info
            return info.get('shortName') or info.get('longName') or stock_code
        except:
            return stock_code
