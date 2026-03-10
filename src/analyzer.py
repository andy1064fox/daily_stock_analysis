# -*- coding: utf-8 -*-
"""
===================================
台股自選股智能分析系统 - AI分析层 (繁體中文特化 🛡️防呆強固版 + N/A修復)
===================================
"""
import json
import logging
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config, get_api_keys_for_model, extra_litellm_params

logger = logging.getLogger(__name__)

# 股票名称映射
STOCK_NAME_MAP = {
    '2330.TW': '台積電',
    '0050.TW': '元大台灣50',
    '0052.TW': '富邦科技',
    '2317.TW': '鴻海',
    '2454.TW': '聯發科',
}

def get_stock_name_multi_source(stock_code: str, context: Optional[Dict] = None, data_manager = None) -> str:
    if context:
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception:
            pass
    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception:
            pass
    return f'股票{stock_code}'

@dataclass
class AnalysisResult:
    code: str
    name: str
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"
    dashboard: Optional[Dict[str, Any]] = None
    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""
    market_snapshot: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code, 'name': self.name, 'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction, 'operation_advice': self.operation_advice,
            'decision_type': self.decision_type, 'confidence_level': self.confidence_level,
            'dashboard': self.dashboard, 'analysis_summary': self.analysis_summary,
            'search_performed': self.search_performed, 'success': self.success,
            'error_message': self.error_message, 'model_used': self.model_used,
            'market_snapshot': self.market_snapshot,
        }

    def get_core_conclusion(self) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            return pos_advice.get('has_position' if has_position else 'no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        emoji_map = {
            '买入': '🟢', '做多進場': '🟢', '加仓': '🟢', '加碼': '🟢',
            '强烈买入': '💚', '強烈做多': '💚', '持有': '🟡', '观望': '⚪',
            '觀望': '⚪', '减仓': '🟠', '減碼': '🟠', '卖出': '🔴',
            '做多出場': '🔴', '强烈卖出': '❌', '強烈出場': '❌',
        }
        advice = self.operation_advice or ''
        if advice in emoji_map: return emoji_map[advice]
        for part in advice.replace('/', '|').split('|'):
            if part.strip() in emoji_map: return emoji_map[part.strip()]
        
        score = self.sentiment_score
        if score >= 80: return '💚'
        if score >= 65: return '🟢'
        if score >= 55: return '🟡'
        if score >= 45: return '⚪'
        if score >= 35: return '🟠'
        return '🔴'

    def get_confidence_stars(self) -> str:
        return {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    SYSTEM_PROMPT = """【系統最高指令】：你是一位專注於趨勢交易的「台股」投資分析師，接下來的所有思考、分析與最終輸出的報告內容，請絕對、務必使用「繁體中文（zh-TW）」撰寫，並使用台灣股市習慣用語（例如：做多進場、做多出場、減碼、停損、本益比等）。

## 核心交易理念
1. 嚴進策略：股價偏離 MA5 超過 5% 時堅決不買入。乖離率 < 2% 為最佳買點。
2. 趨勢交易：多頭排列必須條件為 MA5 > MA10 > MA20。
3. 買點偏好：縮量回踩 MA5 獲得支撐。跌破 MA20 觀望。
4. 風險排查：關注大股東減持、業績預虧。

## 輸出格式
請嚴格按照以下 JSON 格式輸出：
```json
{
    "stock_name": "股票中文名稱",
    "sentiment_score": 50,
    "trend_prediction": "震盪",
    "operation_advice": "觀望",
    "decision_type": "hold",
    "confidence_level": "中",
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句話核心結論",
            "signal_type": "🟡持有觀望",
            "time_sensitivity": "不急",
            "position_advice": {
                "no_position": "空倉者建議",
                "has_position": "持倉者建議"
            }
        },
        "data_perspective": {
            "trend_status": {"ma_alignment": "描述", "is_bullish": false, "trend_score": 50},
            "price_position": {"current_price": 0, "ma5": 0, "ma10": 0, "ma20": 0, "bias_ma5": 0, "bias_status": "安全", "support_level": 0, "resistance_level": 0},
            "volume_analysis": {"volume_ratio": 0, "volume_status": "平量", "turnover_rate": 0, "volume_meaning": "解讀"},
            "chip_structure": {"profit_ratio": 0, "avg_cost": 0, "concentration": 0, "chip_health": "一般"}
        },
        "intelligence": {
            "latest_news": "無",
            "risk_alerts": ["風險"],
            "positive_catalysts": ["利好"],
            "earnings_outlook": "預期",
            "sentiment_summary": "情緒"
        },
        "battle_plan": {
            "sniper_points": {"ideal_buy": "無", "secondary_buy": "無", "stop_loss": "無", "take_profit": "無"},
            "position_strategy": {"suggested_position": "0成", "entry_plan": "無", "risk_control": "無"},
            "action_checklist": ["✅ 檢查項1"]
        }
    },
    "analysis_summary": "摘要"
}
