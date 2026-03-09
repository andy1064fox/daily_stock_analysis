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
            # yfinance 結束日期是不包含的，所以要往後加 1 天
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            end_date_yf = end_dt.strftime("%Y-%m-%d")
            
            logger.info(f"Yfinance 请求历史数据: {stock_code}, {start_date} -> {end_date_yf}")
            ticker = yf.Ticker(stock_code)
            df = ticker.history(start=start_date, end=end_date_yf)
            
            if df.empty:
                logger.warning(f"Yfinance 返回空历史数据: {stock_code}")
                raise DataFetchError(f"Yahoo Finance 返回空数据: {stock_code} (请检查代号是否正确，例如应为 2330.TW)")
                
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Yfinance 历史数据获取异常: {str(e)}")
            raise DataFetchError(f"Yahoo Finance 获取历史数据失败: {e}")

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        df = df.copy()
        
        # 兼容不同版本的 yfinance Date 列
        date_col = 'Date' if 'Date' in df.columns else 'Datetime'
        
        df['date'] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        
        df['open'] = df['Open']
        df['high'] = df['High']
        df['low'] = df['Low']
        df['close'] = df['Close']
        df['volume'] = df['Volume']
        
        # 近似計算成交額
        df['amount'] = df['volume'] * df['close']
        
        df['pct_chg'] = df['close'].pct_change() * 100
        df['pct_chg'] = df['pct_chg'].fillna(0.0)
        
        return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']]

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        try:
            logger.info(f"Yfinance 请求实时行情: {stock_code}")
            ticker = yf.Ticker(stock_code)
            
            # 使用 history 获取最近5天的数据，完全避开 info 的限流报错
            df = ticker.history(period="5d")
            if df.empty:
                logger.warning(f"Yfinance 实时行情为空: {stock_code}")
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
            
            # 尝试获取名称，允许失败以防止卡死
            name = stock_code
            try:
                info = ticker.info
                name = info.get('shortName') or info.get('longName') or stock_code
            except:
                pass

            quote = UnifiedRealtimeQuote(
                stock_code=stock_code,
                name=name,
                current_price=current_price,
                change=change,
                change_pct=change_pct,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
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
            logger.info(f"Yfinance 成功获取行情: {stock_code} = {current_price}")
            return quote
            
        except Exception as e:
            logger.warning(f"Yahoo Finance 获取 {stock_code} 实时行情异常: {e}")
            return None

    def get_stock_name(self, stock_code: str) -> str:
        try:
            info = yf.Ticker(stock_code).info
            return info.get('shortName') or info.get('longName') or stock_code
        except:
            return stock_code
