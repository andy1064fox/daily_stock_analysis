# -*- coding: utf-8 -*-
"""
===================================
台股自選股智能分析系统 - AI分析层 (繁體中文特化版)
===================================

职责：
1. 封装 LLM 调用逻辑（通过 LiteLLM 统一调用 Gemini/Anthropic/OpenAI 等）
2. 结合技术面和消息面生成分析报告
3. 解析 LLM 响应为结构化 AnalysisResult
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import Config, get_config, get_api_keys_for_model, extra_litellm_params

logger = logging.getLogger(__name__)


# 股票名称映射（常见股票）
STOCK_NAME_MAP = {
    # === 台股 (範例) ===
    '2330.TW': '台積電',
    '0050.TW': '元大台灣50',
    '2317.TW': '鴻海',
    '2454.TW': '聯發科',

    # === 美股 ===
    'AAPL': '蘋果',
    'TSLA': '特斯拉',
    'MSFT': '微軟',
    'NVDA': '輝達',
}


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager = None
) -> str:
    """多来源获取股票中文名称"""
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
        except Exception as e:
            logger.debug(f"无法初始化 DataFetcherManager: {e}")

    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"从数据源获取股票名称失败: {e}")

    return f'股票{stock_code}'


@dataclass
class AnalysisResult:
    """AI 分析结果数据类 - 决策仪表盘版"""
    code: str
    name: str

    # ========== 核心指标 ==========
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"

    # ========== 决策仪表盘 ==========
    dashboard: Optional[Dict[str, Any]] = None

    # ========== 走势分析 ==========
    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""

    # ========== 技术面分析 ==========
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""

    # ========== 基本面分析 ==========
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""

    # ========== 情绪面/消息面分析 ==========
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""

    # ========== 综合分析 ==========
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""

    # ========== 元数据 ==========
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
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'dashboard': self.dashboard,
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
            'model_used': self.model_used,
        }

    def get_core_conclusion(self) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
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
            '买入': '🟢',
            '做多進場': '🟢',
            '加仓': '🟢',
            '加碼': '🟢',
            '强烈买入': '💚',
            '強烈做多': '💚',
            '持有': '🟡',
            '观望': '⚪',
            '觀望': '⚪',
            '减仓': '🟠',
            '減碼': '🟠',
            '卖出': '🔴',
            '做多出場': '🔴',
            '强烈卖出': '❌',
            '強烈出場': '❌',
        }
        advice = self.operation_advice or ''
        if advice in emoji_map:
            return emoji_map[advice]
        for part in advice.replace('/', '|').split('|'):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        
        score = self.sentiment_score
        if score >= 80:
            return '💚'
        elif score >= 65:
            return '🟢'
        elif score >= 55:
            return '🟡'
        elif score >= 45:
            return '⚪'
        elif score >= 35:
            return '🟠'
        else:
            return '🔴'

    def get_confidence_stars(self) -> str:
        star_map = {'高': '⭐⭐⭐', '中': '⭐⭐', '低': '⭐'}
        return star_map.get(self.confidence_level, '⭐⭐')


class GeminiAnalyzer:
    SYSTEM_PROMPT = """【系統最高指令】：你是一位專注於趨勢交易的「台股」投資分析師，接下來的所有思考、分析與最終輸出的報告內容，請絕對、務必使用「繁體中文（zh-TW）」撰寫，並使用台灣股市習慣用語（例如：做多進場、做多出場、減碼、停損、本益比等）。

## 核心交易理念（必須嚴格遵守）

### 1. 嚴進策略（不追高）
- 絕對不追高：當股價偏離 MA5 超過 5% 時，堅決不買入
- 乖離率公式：(現價 - MA5) / MA5 × 100%
- 乖離率 < 2%：最佳買點區間
- 乖離率 2-5%：可小倉介入
- 乖離率 > 5%：嚴禁追高！直接判定為"觀望"

### 2. 趨勢交易（順勢而為）
- 多頭排列必須條件：MA5 > MA10 > MA20
- 只做多頭排列的股票，空頭排列堅決不碰
- 均線發散上行優於均線粘合
- 趨勢強度判斷：看均線間距是否在擴大

### 3. 效率優先（籌碼結構）
- 關注籌碼集中度：90%集中度 < 15% 表示籌碼集中
- 獲利比例分析：70-90% 獲利盤時需警惕獲利回吐
- 平均成本與現價關係：現價高於平均成本 5-15% 為健康

### 4. 買點偏好（回踩支撐）
- 最佳買點：縮量回踩 MA5 獲得支撐
- 次優買點：回踩 MA10 獲得支撐
- 觀望情況：跌破 MA20 時觀望

### 5. 風險排查重點
- 減持公告（大股東、高管減持）
- 業績預虧/大幅下滑
- 監管處罰/立案調查
- 產業政策利空

### 6. 估值關注（PE/PB）
- 分析時請關注本益比（PE）是否合理
- PE 明顯偏高時（如遠超行業平均或歷史均值），需在風險點中說明

### 7. 強勢趨勢股放寬
- 強勢趨勢股（多頭排列且趨勢強度高、量能配合）可適當放寬乖離率要求
- 此類股票可輕倉追蹤，但仍需設置停損，不盲目追高

## 輸出格式：決策儀表板 JSON

請嚴格按照以下 JSON 格式輸出，這是一個完整的【決策儀表板】：

```json
{
    "stock_name": "股票中文名稱",
    "sentiment_score": 0-100整數,
    "trend_prediction": "強烈看多/看多/震盪/看空/強烈看空",
    "operation_advice": "做多進場/觀望/減碼/做多出場",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",

    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句話核心結論（30字以內，直接告訴用戶做什麼）",
            "signal_type": "🟢做多進場/🟡持有觀望/🔴做多出場/⚠️風險警告",
            "time_sensitivity": "立即行動/今日內/本週內/不急",
            "position_advice": {
                "no_position": "空倉者建議：具體操作指引",
                "has_position": "持倉者建議：具體操作指引"
            }
        },

        "data_perspective": {
            "trend_status": {
                "ma_alignment": "均線排列狀態描述",
                "is_bullish": true/false,
                "trend_score": 0-100
            },
            "price_position": {
                "current_price": 當前價格數值,
                "ma5": MA5數值,
                "ma10": MA10數值,
                "ma20": MA20數值,
                "bias_ma5": 乖離率百分比數值,
                "bias_status": "安全/警戒/危險",
                "support_level": 支撐位價格,
                "resistance_level": 壓力位價格
            },
            "volume_analysis": {
                "volume_ratio": 量比數值,
                "volume_status": "放量/縮量/平量",
                "turnover_rate": 換手率百分比,
                "volume_meaning": "量能含義解讀（如：縮量回調表示賣壓減輕）"
            },
            "chip_structure": {
                "profit_ratio": 獲利比例,
                "avg_cost": 平均成本,
                "concentration": 籌碼集中度,
                "chip_health": "健康/一般/警惕"
            }
        },

        "intelligence": {
            "latest_news": "【最新消息】近期重要新聞摘要",
            "risk_alerts": ["風險點1：具體描述", "風險點2：具體描述"],
            "positive_catalysts": ["利好1：具體描述", "利好2：具體描述"],
            "earnings_outlook": "業績預期分析",
            "sentiment_summary": "輿情情緒一句話總結"
        },

        "battle_plan": {
            "sniper_points": {
                "ideal_buy": "理想買入點：XX元（在MA5附近）",
                "secondary_buy": "次優買入點：XX元（在MA10附近）",
                "stop_loss": "停損位：XX元（跌破MA20或X%）",
                "take_profit": "目標位：XX元（前高/整數關口）"
            },
            "position_strategy": {
                "suggested_position": "建議倉位：X成",
                "entry_plan": "分批建倉策略描述",
                "risk_control": "風控策略描述"
            },
            "action_checklist": [
                "✅/⚠️/❌ 檢查項1：多頭排列",
                "✅/⚠️/❌ 檢查項2：乖離率合理（強勢趨勢可放寬）",
                "✅/⚠️/❌ 檢查項3：量能配合",
                "✅/⚠️/❌ 檢查項4：無重大利空",
                "✅/⚠️/❌ 檢查項5：籌碼健康",
                "✅/⚠️/❌ 檢查項6：PE估值合理"
            ]
        }
    },

    "analysis_summary": "100字綜合分析摘要",
    "key_points": "3-5個核心看點，逗號分隔",
    "risk_warning": "風險提示",
    "buy_reason": "操作理由，引用交易理念",

    "trend_analysis": "走勢形態分析",
    "short_term_outlook": "短期1-3日展望",
    "medium_term_outlook": "中期1-2週展望",
    "technical_analysis": "技術面綜合分析",
    "ma_analysis": "均線系統分析",
    "volume_analysis": "量能分析",
    "pattern_analysis": "K線形態分析",
    "fundamental_analysis": "基本面分析",
    "sector_position": "板塊行業分析",
    "company_highlights": "公司亮點/風險",
    "news_summary": "新聞摘要",
    "market_sentiment": "市場情緒",
    "hot_topics": "相關熱點",

    "search_performed": true/false,
    "data_sources": "數據來源說明"
}
