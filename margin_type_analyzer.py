#!/usr/bin/env python3
"""
AI Margin Type Selection System

Analyzes trading strategy and market conditions to recommend
optimal margin type (ISOLATED vs CROSS) for each trade.
"""

import asyncio
import ccxt
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os

class MarginType(Enum):
    """Margin type options"""
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"

@dataclass
class MarginAnalysis:
    """Margin type analysis result"""
    recommended_type: MarginType
    confidence: float
    reasoning: List[str]
    risk_assessment: str
    capital_efficiency: str
    safety_level: str
    alternative_option: Optional[MarginType]
    trade_context: Dict[str, Any]

class MarginTypeOptimizer:
    """AI system to optimize margin type selection"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Strategy context
        self.current_positions = []
        self.portfolio_balance = 0.0
        self.risk_tolerance = "MODERATE"  # LOW, MODERATE, HIGH
        self.experience_level = "INTERMEDIATE"  # BEGINNER, INTERMEDIATE, EXPERT
        
        print("🎯 AI Margin Type Optimizer")
        print("=" * 35)
        print("Available Options:")
        print("   • ISOLATED: Risk confined per position")
        print("   • CROSS: Shared margin across positions")
    
    async def analyze_optimal_margin_type(self, 
                                        trade_data: Dict[str, Any],
                                        portfolio_data: Dict[str, Any]) -> MarginAnalysis:
        """Analyze and recommend optimal margin type"""
        try:
            # Comprehensive analysis factors
            analysis_factors = self._gather_analysis_factors(trade_data, portfolio_data)
            
            # Get AI recommendation
            ai_recommendation = await self._get_ai_margin_recommendation(analysis_factors)
            
            # Apply rule-based validation
            final_recommendation = self._validate_and_finalize(ai_recommendation, analysis_factors)
            
            return final_recommendation
            
        except Exception as e:
            print(f"❌ Error analyzing margin type: {e}")
            # Safe fallback to ISOLATED
            return self._create_safe_fallback()
    
    def _gather_analysis_factors(self, trade_data: Dict[str, Any], 
                                portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive factors for margin analysis"""
        
        # Trade characteristics
        setup_quality = trade_data.get('confidence_score', 70)
        capital_usage = trade_data.get('capital_usage_pct', 0.5)
        leverage = trade_data.get('leverage', 10)
        symbol = trade_data.get('symbol', 'BTC/USDT')
        direction = trade_data.get('direction', 'BUY')
        
        # Portfolio state
        total_balance = portfolio_data.get('total_balance', 1000)
        available_balance = portfolio_data.get('available_balance', 1000)
        current_positions_count = len(portfolio_data.get('positions', []))
        portfolio_pnl = portfolio_data.get('unrealized_pnl', 0)
        
        # Risk metrics
        position_risk_pct = capital_usage * leverage * 0.02  # Estimated 2% stop loss
        portfolio_risk_pct = position_risk_pct
        
        # Market conditions
        volatility = abs(trade_data.get('price_change_24h', 0))
        market_regime = trade_data.get('market_regime', 'SIDEWAYS')
        
        # Strategy context
        is_high_conviction = setup_quality >= 85 and len(trade_data.get('confluences', [])) >= 4
        is_large_position = capital_usage >= 0.7
        is_high_leverage = leverage >= 15
        
        return {
            # Trade characteristics
            'setup_quality': setup_quality,
            'capital_usage': capital_usage,
            'leverage': leverage,
            'symbol': symbol,
            'direction': direction,
            'is_high_conviction': is_high_conviction,
            'is_large_position': is_large_position,
            'is_high_leverage': is_high_leverage,
            
            # Portfolio state
            'total_balance': total_balance,
            'available_balance': available_balance,
            'current_positions_count': current_positions_count,
            'portfolio_pnl': portfolio_pnl,
            'capital_utilization': (total_balance - available_balance) / total_balance,
            
            # Risk metrics
            'position_risk_pct': position_risk_pct,
            'portfolio_risk_pct': portfolio_risk_pct,
            'liquidation_distance': 1 / leverage * 0.9,  # Approximate
            
            # Market conditions
            'volatility': volatility,
            'market_regime': market_regime,
            'is_high_volatility': volatility > 5,
            
            # Context flags
            'has_existing_positions': current_positions_count > 0,
            'portfolio_in_profit': portfolio_pnl > 0,
            'capital_heavily_used': capital_usage > 0.8
        }
    
    async def _get_ai_margin_recommendation(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI recommendation for margin type"""
        
        context = f"""
        MARGIN TYPE OPTIMIZATION ANALYSIS
        
        Trade Details:
        - Symbol: {factors['symbol']} {factors['direction']}
        - Setup Quality: {factors['setup_quality']:.1f}%
        - Capital Usage: {factors['capital_usage']:.0%}
        - Leverage: {factors['leverage']}x
        - High Conviction: {factors['is_high_conviction']}
        - Large Position: {factors['is_large_position']}
        - High Leverage: {factors['is_high_leverage']}
        
        Portfolio State:
        - Total Balance: ${factors['total_balance']:,.2f}
        - Available: ${factors['available_balance']:,.2f}
        - Current Positions: {factors['current_positions_count']}
        - Portfolio P&L: ${factors['portfolio_pnl']:+,.2f}
        - Capital Utilization: {factors['capital_utilization']:.1%}
        
        Risk Assessment:
        - Position Risk: {factors['position_risk_pct']:.1%}
        - Liquidation Distance: {factors['liquidation_distance']:.1%}
        - Market Volatility: {factors['volatility']:.1f}%
        - Market Regime: {factors['market_regime']}
        
        MARGIN TYPE DECISION FACTORS:
        
        ISOLATED MARGIN - Best for:
        ✅ Risk Control: Each position isolated, limited downside
        ✅ High Leverage: Safer with high leverage (15x+)
        ✅ Large Positions: When using 70%+ of capital
        ✅ High Volatility: Uncertain market conditions
        ✅ Single Focus: Trading one position at a time
        ✅ Learning: Beginner-friendly risk management
        
        CROSS MARGIN - Best for:
        ✅ Capital Efficiency: Use full balance for margin
        ✅ Multiple Positions: Portfolio of positions
        ✅ Lower Leverage: 5x-10x range with multiple trades
        ✅ Stable Markets: Lower volatility conditions
        ✅ Hedging: Offsetting positions (long + short)
        ✅ Advanced Strategies: Complex portfolio management
        
        CRITICAL CONSIDERATIONS:
        1. Risk Tolerance: How much total account at risk?
        2. Position Size: Larger positions favor ISOLATED
        3. Portfolio Complexity: Multiple positions favor CROSS
        4. Market Conditions: High volatility favors ISOLATED
        5. Experience Level: Beginners should use ISOLATED
        """
        
        prompt = f"""
        {context}
        
        Based on this comprehensive analysis, determine the optimal margin type.
        
        Consider:
        1. Risk management priorities
        2. Capital efficiency needs
        3. Portfolio complexity
        4. Market volatility
        5. Position characteristics
        
        Analyze the tradeoffs between safety (ISOLATED) vs efficiency (CROSS).
        
        Respond ONLY with valid JSON:
        {{
            "recommended_margin_type": "ISOLATED" | "CROSS",
            "confidence_score": float (0-100),
            "primary_reasoning": [
                "key reason 1",
                "key reason 2", 
                "key reason 3"
            ],
            "risk_assessment": "LOW" | "MEDIUM" | "HIGH",
            "capital_efficiency": "LOW" | "MEDIUM" | "HIGH",
            "safety_level": "LOW" | "MEDIUM" | "HIGH",
            "alternative_consideration": "ISOLATED" | "CROSS" | null,
            "key_tradeoffs": [
                "tradeoff 1",
                "tradeoff 2"
            ],
            "warnings": [
                "warning 1",
                "warning 2"
            ]
        }}
        """
        
        # Try AI analysis
        if self.openai_api_key:
            try:
                import openai
                openai.api_key = self.openai_api_key
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in futures margin trading with deep knowledge of risk management. Analyze margin type selection carefully. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                
                response_text = response.choices[0].message.content.strip()
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                return json.loads(response_text)
                
            except Exception as e:
                print(f"⚠️ AI margin analysis failed: {e}")
        
        # Fallback rule-based analysis
        return self._rule_based_margin_analysis(factors)
    
    def _rule_based_margin_analysis(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based margin type recommendation"""
        
        # Decision logic based on key factors
        isolated_score = 0
        cross_score = 0
        
        # Factor 1: Position size and risk
        if factors['is_large_position']:
            isolated_score += 3  # Large positions safer with ISOLATED
        else:
            cross_score += 1
        
        # Factor 2: Leverage level
        if factors['is_high_leverage']:
            isolated_score += 3  # High leverage safer with ISOLATED
        elif factors['leverage'] <= 10:
            cross_score += 2  # Lower leverage can use CROSS
        
        # Factor 3: Portfolio complexity
        if factors['current_positions_count'] == 0:
            isolated_score += 2  # Single position favors ISOLATED
        elif factors['current_positions_count'] >= 2:
            cross_score += 3  # Multiple positions favor CROSS
        
        # Factor 4: Market volatility
        if factors['is_high_volatility']:
            isolated_score += 2  # High volatility favors ISOLATED
        else:
            cross_score += 1
        
        # Factor 5: Setup conviction
        if factors['is_high_conviction']:
            isolated_score += 1  # High conviction can use ISOLATED safely
            cross_score += 1     # But also works with CROSS
        
        # Factor 6: Capital utilization
        if factors['capital_heavily_used']:
            isolated_score += 2  # Heavy capital use favors ISOLATED
        
        # Determine recommendation
        if isolated_score > cross_score:
            recommended = "ISOLATED"
            confidence = min(80, 60 + (isolated_score - cross_score) * 5)
            risk_level = "MEDIUM"
            safety_level = "HIGH"
            efficiency = "MEDIUM"
        else:
            recommended = "CROSS"
            confidence = min(80, 60 + (cross_score - isolated_score) * 5)
            risk_level = "MEDIUM"
            safety_level = "MEDIUM"
            efficiency = "HIGH"
        
        reasoning = []
        if factors['is_large_position']:
            reasoning.append(f"Large position ({factors['capital_usage']:.0%}) favors risk isolation")
        if factors['is_high_leverage']:
            reasoning.append(f"High leverage ({factors['leverage']}x) needs isolated risk")
        if factors['current_positions_count'] > 1:
            reasoning.append(f"Multiple positions ({factors['current_positions_count']}) benefit from shared margin")
        if factors['is_high_volatility']:
            reasoning.append(f"High volatility ({factors['volatility']:.1f}%) suggests isolated margin")
        
        return {
            "recommended_margin_type": recommended,
            "confidence_score": confidence,
            "primary_reasoning": reasoning[:3] if reasoning else ["Rule-based analysis applied"],
            "risk_assessment": risk_level,
            "capital_efficiency": efficiency,
            "safety_level": safety_level,
            "alternative_consideration": "CROSS" if recommended == "ISOLATED" else "ISOLATED",
            "key_tradeoffs": [
                "Safety vs Capital Efficiency",
                "Risk Isolation vs Portfolio Flexibility"
            ],
            "warnings": [
                "Monitor position closely regardless of margin type",
                "Consider switching margin type as conditions change"
            ]
        }
    
    def _validate_and_finalize(self, ai_recommendation: Dict[str, Any],
                             factors: Dict[str, Any]) -> MarginAnalysis:
        """Validate AI recommendation and create final analysis"""
        
        recommended_type = MarginType(ai_recommendation.get('recommended_margin_type', 'ISOLATED'))
        confidence = ai_recommendation.get('confidence_score', 70)
        
        # Safety overrides
        if factors['is_high_leverage'] and factors['is_large_position']:
            # Force ISOLATED for high risk combinations
            if recommended_type == MarginType.CROSS:
                print("⚠️ Safety override: Forcing ISOLATED for high leverage + large position")
                recommended_type = MarginType.ISOLATED
                confidence = max(85, confidence)
        
        # Create comprehensive analysis
        return MarginAnalysis(
            recommended_type=recommended_type,
            confidence=confidence,
            reasoning=ai_recommendation.get('primary_reasoning', []),
            risk_assessment=ai_recommendation.get('risk_assessment', 'MEDIUM'),
            capital_efficiency=ai_recommendation.get('capital_efficiency', 'MEDIUM'),
            safety_level=ai_recommendation.get('safety_level', 'MEDIUM'),
            alternative_option=MarginType(ai_recommendation['alternative_consideration']) if ai_recommendation.get('alternative_consideration') else None,
            trade_context=factors
        )
    
    def _create_safe_fallback(self) -> MarginAnalysis:
        """Create safe fallback recommendation"""
        return MarginAnalysis(
            recommended_type=MarginType.ISOLATED,
            confidence=80.0,
            reasoning=["Safe default due to analysis error", "ISOLATED provides risk isolation", "Recommended for uncertain conditions"],
            risk_assessment="LOW",
            capital_efficiency="MEDIUM",
            safety_level="HIGH",
            alternative_option=MarginType.CROSS,
            trade_context={}
        )
    
    def display_margin_analysis(self, analysis: MarginAnalysis, trade_data: Dict[str, Any]):
        """Display comprehensive margin analysis"""
        print(f"\n🎯 MARGIN TYPE ANALYSIS")
        print("=" * 35)
        print(f"📊 Trade: {trade_data.get('symbol', 'N/A')} {trade_data.get('direction', 'N/A')}")
        print(f"💰 Capital: {trade_data.get('capital_usage_pct', 0):.0%}")
        print(f"📈 Leverage: {trade_data.get('leverage', 0)}x")
        
        print(f"\n🏆 RECOMMENDED: {analysis.recommended_type.value}")
        print(f"🎲 Confidence: {analysis.confidence:.1f}%")
        
        print(f"\n📊 ANALYSIS METRICS:")
        print(f"   Risk Level: {analysis.risk_assessment}")
        print(f"   Capital Efficiency: {analysis.capital_efficiency}")
        print(f"   Safety Level: {analysis.safety_level}")
        
        if analysis.alternative_option:
            print(f"   Alternative: {analysis.alternative_option.value}")
        
        print(f"\n🧠 KEY REASONING:")
        for reason in analysis.reasoning:
            print(f"   • {reason}")
        
        # Show comparison
        print(f"\n⚖️  MARGIN TYPE COMPARISON:")
        print(f"   ISOLATED MARGIN:")
        print(f"     ✅ Risk confined per position")
        print(f"     ✅ Safe for high leverage")
        print(f"     ✅ Beginner-friendly")
        print(f"     ❌ Lower capital efficiency")
        
        print(f"   CROSS MARGIN:")
        print(f"     ✅ Higher capital efficiency")
        print(f"     ✅ Good for multiple positions")
        print(f"     ✅ Portfolio netting benefits")
        print(f"     ❌ Higher total account risk")

class MarginTypeRecommendationEngine:
    """Complete margin type recommendation engine"""
    
    def __init__(self):
        self.optimizer = MarginTypeOptimizer()
        
        # Trading scenarios for analysis
        self.scenarios = {
            'conservative_single': {
                'name': 'Conservative Single Position',
                'trade_data': {
                    'symbol': 'BTC/USDT',
                    'direction': 'BUY',
                    'confidence_score': 78.0,
                    'capital_usage_pct': 0.6,
                    'leverage': 8,
                    'confluences': ['BOS', 'FVG', 'EMA'],
                    'price_change_24h': 1.2,
                    'market_regime': 'BULL'
                },
                'portfolio_data': {
                    'total_balance': 1000.0,
                    'available_balance': 1000.0,
                    'positions': [],
                    'unrealized_pnl': 0.0
                }
            },
            'aggressive_high_conviction': {
                'name': 'Aggressive High Conviction',
                'trade_data': {
                    'symbol': 'ETH/USDT',
                    'direction': 'BUY',
                    'confidence_score': 92.0,
                    'capital_usage_pct': 0.85,
                    'leverage': 18,
                    'confluences': ['BOS', 'FVG', 'EMA', 'RSI', 'Volume'],
                    'price_change_24h': 3.5,
                    'market_regime': 'BULL'
                },
                'portfolio_data': {
                    'total_balance': 1000.0,
                    'available_balance': 1000.0,
                    'positions': [],
                    'unrealized_pnl': 0.0
                }
            },
            'multiple_positions': {
                'name': 'Multiple Position Portfolio',
                'trade_data': {
                    'symbol': 'ADA/USDT',
                    'direction': 'SELL',
                    'confidence_score': 75.0,
                    'capital_usage_pct': 0.33,
                    'leverage': 6,
                    'confluences': ['BOS', 'EMA', 'RSI'],
                    'price_change_24h': -1.8,
                    'market_regime': 'BEAR'
                },
                'portfolio_data': {
                    'total_balance': 1000.0,
                    'available_balance': 600.0,
                    'positions': [
                        {'symbol': 'BTC/USDT', 'side': 'long'},
                        {'symbol': 'ETH/USDT', 'side': 'short'}
                    ],
                    'unrealized_pnl': 50.0
                }
            }
        }
    
    async def analyze_all_scenarios(self):
        """Analyze all trading scenarios"""
        print("🔍 Analyzing Margin Type for Different Scenarios")
        print("=" * 55)
        
        for scenario_key, scenario in self.scenarios.items():
            print(f"\n📋 Scenario: {scenario['name']}")
            print("-" * 40)
            
            analysis = await self.optimizer.analyze_optimal_margin_type(
                scenario['trade_data'],
                scenario['portfolio_data']
            )
            
            self.optimizer.display_margin_analysis(analysis, scenario['trade_data'])
    
    def provide_general_recommendations(self):
        """Provide general margin type recommendations"""
        print(f"\n📋 GENERAL MARGIN TYPE RECOMMENDATIONS")
        print("=" * 45)
        
        recommendations = [
            {
                'scenario': 'Beginner Trader',
                'recommendation': 'ISOLATED',
                'reason': 'Learn risk management with confined losses'
            },
            {
                'scenario': 'High Leverage (15x+)',
                'recommendation': 'ISOLATED', 
                'reason': 'Limit liquidation risk to single position'
            },
            {
                'scenario': 'Large Position (70%+ capital)',
                'recommendation': 'ISOLATED',
                'reason': 'Protect remaining capital from major loss'
            },
            {
                'scenario': 'Multiple Positions',
                'recommendation': 'CROSS',
                'reason': 'Efficient capital use and portfolio netting'
            },
            {
                'scenario': 'Low Leverage (5x-10x)',
                'recommendation': 'CROSS',
                'reason': 'Safe leverage allows shared margin benefits'
            },
            {
                'scenario': 'High Volatility Markets',
                'recommendation': 'ISOLATED',
                'reason': 'Unpredictable moves favor risk isolation'
            },
            {
                'scenario': 'Stable Low Volatility',
                'recommendation': 'CROSS',
                'reason': 'Predictable conditions allow shared margin'
            },
            {
                'scenario': 'Hedged Portfolio',
                'recommendation': 'CROSS',
                'reason': 'Long/short positions can offset each other'
            }
        ]
        
        for rec in recommendations:
            print(f"📌 {rec['scenario']}: {rec['recommendation']}")
            print(f"   → {rec['reason']}")

async def main():
    """Main function to demonstrate margin type analysis"""
    engine = MarginTypeRecommendationEngine()
    
    # Analyze all scenarios
    await engine.analyze_all_scenarios()
    
    # Provide general recommendations
    engine.provide_general_recommendations()
    
    print(f"\n🎯 CONCLUSION FOR YOUR STRATEGY:")
    print("=" * 35)
    print("Based on your AI-enhanced futures trading strategy:")
    print("")
    print("✅ RECOMMENDED: **ISOLATED MARGIN**")
    print("")
    print("🔑 Key Reasons:")
    print("   • AI determines high leverage (up to 20x)")
    print("   • Large capital usage (50-85%)")
    print("   • Single position focus strategy")
    print("   • Risk isolation per trade")
    print("   • Safer for automated trading")
    print("")
    print("⚠️ When to Consider CROSS:")
    print("   • Multiple smaller positions")
    print("   • Lower leverage (5x-10x)")
    print("   • Portfolio hedging strategies")
    print("   • Expert manual trading")

if __name__ == "__main__":
    asyncio.run(main())