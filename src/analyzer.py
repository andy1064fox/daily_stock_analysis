# -*- coding: utf-8 -*-
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
"""

    def __init__(self, api_key: Optional[str] = None):
        self._router = None
        self._litellm_available = False
        self._init_litellm()

    def _has_channel_config(self, config: Config) -> bool:
        return bool(config.llm_model_list) and not all(
            e.get('model_name', '').startswith('__legacy_') for e in config.llm_model_list
        )

    def _init_litellm(self) -> None:
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model: return
        self._litellm_available = True

        if self._has_channel_config(config):
            self._router = Router(model_list=config.llm_model_list, routing_strategy="simple-shuffle", num_retries=2)
            return

        keys = get_api_keys_for_model(litellm_model, config)
        if len(keys) > 1:
            extra_params = extra_litellm_params(litellm_model, config)
            self._router = Router(
                model_list=[{"model_name": litellm_model, "litellm_params": {"model": litellm_model, "api_key": k, **extra_params}} for k in keys],
                routing_strategy="simple-shuffle", num_retries=2
            )

    def is_available(self) -> bool:
        return self._router is not None or self._litellm_available

    def _call_litellm(self, prompt: str, generation_config: dict) -> Tuple[str, str]:
        config = get_config()
        max_tokens = generation_config.get('max_output_tokens') or 8192
        temperature = generation_config.get('temperature', 0.7)
        models_to_try = [m for m in [config.litellm_model] + (config.litellm_fallback_models or []) if m]
        use_channel_router = self._has_channel_config(config)
        last_error = None
        
        for model in models_to_try:
            try:
                call_kwargs = {
                    "model": model,
                    "messages": [{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    "temperature": temperature, "max_tokens": max_tokens
                }
                extra = get_thinking_extra_body(model.split("/")[-1] if "/" in model else model)
                if extra: call_kwargs["extra_body"] = extra

                if use_channel_router and self._router:
                    response = self._router.completion(**call_kwargs)
                elif self._router and model == config.litellm_model:
                    response = self._router.completion(**call_kwargs)
                else:
                    keys = get_api_keys_for_model(model, config)
                    if keys: call_kwargs["api_key"] = keys[0]
                    call_kwargs.update(extra_litellm_params(model, config))
                    response = litellm.completion(**call_kwargs)

                if response and response.choices and response.choices[0].message.content:
                    return (response.choices[0].message.content, model)
            except Exception as e:
                last_error = e
                continue
        raise Exception(f"All LLM models failed. Last error: {last_error}")

    def generate_text(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Optional[str]:
        try:
            res = self._call_litellm(prompt, {"max_tokens": max_tokens, "temperature": temperature})
            return res[0] if isinstance(res, tuple) else res
        except Exception:
            return None

    def analyze(self, context: Dict[str, Any], news_context: Optional[str] = None) -> AnalysisResult:
        code = context.get('code', 'Unknown')
        config = get_config()
        
        if config.gemini_request_delay > 0:
            time.sleep(config.gemini_request_delay)
        
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            name = context.get('realtime', {}).get('name') or STOCK_NAME_MAP.get(code, f'股票{code}')
        
        if not self.is_available():
            return AnalysisResult(code=code, name=name, sentiment_score=50, trend_prediction='震盪', operation_advice='觀望', analysis_summary='AI 分析未啟用', success=False)
        
        try:
            prompt = self._format_prompt(context, name, news_context)
            response_text, model_used = self._call_litellm(prompt, {"temperature": config.gemini_temperature, "max_output_tokens": 8192})
            result = self._parse_response(response_text, code, name)
            result.raw_response = response_text
            result.search_performed = bool(news_context)
            result.market_snapshot = self._build_market_snapshot(context)
            result.model_used = model_used
            return result
        except Exception as e:
            return AnalysisResult(code=code, name=name, sentiment_score=50, trend_prediction='震盪', operation_advice='觀望', analysis_summary=f'分析出錯: {str(e)[:100]}', success=False, error_message=str(e))
    
    def _format_prompt(self, context: Dict[str, Any], name: str, news_context: Optional[str] = None) -> str:
        code = context.get('code', 'Unknown')
        stock_name = STOCK_NAME_MAP.get(code, name)
        today = context.get('today', {})
        
        prompt = f"# 決策儀表板分析請求\n\n## 📊 基礎資訊\n代碼: {code} | 名稱: {stock_name}\n"
        prompt += f"收盤價: {today.get('close', 'N/A')} | MA5: {today.get('ma5', 'N/A')} | MA20: {today.get('ma20', 'N/A')}\n"
        
        if 'realtime' in context:
            prompt += f"量比: {context['realtime'].get('volume_ratio', 'N/A')} | 本益比: {context['realtime'].get('pe_ratio', 'N/A')}\n"
            
        prompt += "\n## 📰 輿情情報\n" + (news_context if news_context else "無新聞")
        prompt += "\n請生成完整 JSON 決策儀表板，必須使用繁體中文！"
        return prompt
    
    def _format_price(self, value: Optional[float]) -> str:
        return f"{float(value):.2f}" if value is not None else 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"date": context.get('date', '未知'), "close": self._format_price(context.get('today', {}).get('close'))}

    def _parse_response(self, response_text: str, code: str, name: str) -> AnalysisResult:
        try:
            cleaned = response_text.split('```json')[1].split('```')[0] if '```json' in response_text else response_text.replace('```', '')
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = repair_json(cleaned[start:end])
                data = json.loads(json_str) if isinstance(json_str, str) else json_str
                if not isinstance(data, dict): raise ValueError("Parsed JSON is not a dict")
                
                op = data.get('operation_advice', '觀望')
                if op in ['買入', '加仓', '做多進場', '加碼']: dt = 'buy'
                elif op in ['賣出', '减仓', '做多出場', '減碼', '停損']: dt = 'sell'
                else: dt = 'hold'
                
                score = data.get('sentiment_score', 50)
                try: score = int(score) if score is not None else 50
                except: score = 50
                
                return AnalysisResult(
                    code=code, name=data.get('stock_name', name), sentiment_score=score,
                    trend_prediction=data.get('trend_prediction', '震盪'), operation_advice=op,
                    decision_type=dt, confidence_level=data.get('confidence_level', '中'),
                    dashboard=data.get('dashboard'), analysis_summary=data.get('analysis_summary', '分析完成'), success=True
                )
        except Exception:
            pass
        
        text_lower = response_text.lower()
        pos = sum(1 for kw in ['看多', '進場', 'buy'] if kw in text_lower)
        neg = sum(1 for kw in ['看空', '出場', 'sell'] if kw in text_lower)
        if pos > neg: return AnalysisResult(code=code, name=name, sentiment_score=65, trend_prediction='看多', operation_advice='做多進場', decision_type='buy')
        if neg > pos: return AnalysisResult(code=code, name=name, sentiment_score=35, trend_prediction='看空', operation_advice='做多出場', decision_type='sell')
        return AnalysisResult(code=code, name=name, sentiment_score=50, trend_prediction='震盪', operation_advice='觀望', decision_type='hold')

   def batch_analyze(self, contexts: List[Dict[str, Any]], delay_between: float = 65.0) -> List[AnalysisResult]:
        """批量分析，加入強制 65 秒冷卻以防 Gemini 速率限制 (Rate Limit)"""
        results = []
        for i, context in enumerate(contexts):
            if i > 0:
                logger.info(f"防限流保護：等待 {delay_between} 秒後分析下一檔...")
                time.sleep(delay_between)
            
            # 加入重試機制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.analyze(context)
                    if result.error_message and "RateLimitError" in result.error_message:
                        raise Exception("RateLimitError")
                    results.append(result)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 30  # 如果失敗，多等30秒
                        logger.warning(f"分析失敗，{wait_time}秒後重試... 錯誤: {e}")
                        time.sleep(wait_time)
                    else:
                        results.append(self.analyze(context))
        return results

def get_analyzer() -> GeminiAnalyzer:
    return GeminiAnalyzer()
