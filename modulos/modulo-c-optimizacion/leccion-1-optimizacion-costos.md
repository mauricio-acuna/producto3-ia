# 💰 Lección 1: Optimización de Costos en LLMs

## 🎯 Objetivos de la Lección

Al finalizar esta lección, serás capaz de:
- Entender la estructura de costos en LLMs (tokens, modelos, requests)
- Implementar estrategias de optimización de tokens
- Configurar sistemas de caching inteligente
- Seleccionar modelos óptimos por caso de uso
- Monitorear y controlar presupuestos en tiempo real
- Implementar rate limiting y quotas

## 📊 Anatomía de los Costos en LLMs

### 1. Estructura de Costos por Proveedor

```python
from typing import Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    COHERE = "cohere"

@dataclass
class ModelPricing:
    model_name: str
    provider: ModelProvider
    input_price_per_1k: float  # USD por 1K tokens de entrada
    output_price_per_1k: float  # USD por 1K tokens de salida
    context_window: int
    capabilities: list
    performance_tier: str  # "basic", "advanced", "premium"

# Precios actualizados 2025 (ejemplo)
MODEL_PRICING = {
    # OpenAI GPT-4 Family
    "gpt-4-turbo": ModelPricing(
        model_name="gpt-4-turbo",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
        context_window=128000,
        capabilities=["chat", "function_calling", "vision"],
        performance_tier="premium"
    ),
    "gpt-4": ModelPricing(
        model_name="gpt-4",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.03,
        output_price_per_1k=0.06,
        context_window=8192,
        capabilities=["chat", "function_calling"],
        performance_tier="premium"
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.0015,
        output_price_per_1k=0.002,
        context_window=16385,
        capabilities=["chat", "function_calling"],
        performance_tier="advanced"
    ),
    
    # Anthropic Claude Family
    "claude-3-opus": ModelPricing(
        model_name="claude-3-opus",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.015,
        output_price_per_1k=0.075,
        context_window=200000,
        capabilities=["chat", "analysis", "coding"],
        performance_tier="premium"
    ),
    "claude-3-sonnet": ModelPricing(
        model_name="claude-3-sonnet",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        context_window=200000,
        capabilities=["chat", "analysis"],
        performance_tier="advanced"
    ),
    "claude-3-haiku": ModelPricing(
        model_name="claude-3-haiku",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.00125,
        context_window=200000,
        capabilities=["chat", "simple_tasks"],
        performance_tier="basic"
    )
}

class CostCalculator:
    """Calculadora de costos para LLMs"""
    
    def __init__(self):
        self.pricing_data = MODEL_PRICING
    
    def calculate_request_cost(self, 
                             model_name: str, 
                             input_tokens: int, 
                             output_tokens: int) -> Dict[str, Any]:
        """Calcular costo de una request específica"""
        
        if model_name not in self.pricing_data:
            raise ValueError(f"Model {model_name} not found in pricing data")
        
        pricing = self.pricing_data[model_name]
        
        # Calcular costos
        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
        total_cost = input_cost + output_cost
        
        return {
            'model': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'input_cost_usd': round(input_cost, 6),
            'output_cost_usd': round(output_cost, 6),
            'total_cost_usd': round(total_cost, 6),
            'cost_per_token': round(total_cost / (input_tokens + output_tokens), 8),
            'provider': pricing.provider.value
        }
    
    def compare_models(self, 
                      input_tokens: int, 
                      output_tokens: int,
                      required_capabilities: list = None) -> Dict[str, Any]:
        """Comparar costos entre diferentes modelos"""
        
        results = []
        
        for model_name, pricing in self.pricing_data.items():
            # Filtrar por capacidades si se especifican
            if required_capabilities:
                if not all(cap in pricing.capabilities for cap in required_capabilities):
                    continue
            
            cost_info = self.calculate_request_cost(model_name, input_tokens, output_tokens)
            cost_info['performance_tier'] = pricing.performance_tier
            cost_info['context_window'] = pricing.context_window
            cost_info['capabilities'] = pricing.capabilities
            
            results.append(cost_info)
        
        # Ordenar por costo
        results.sort(key=lambda x: x['total_cost_usd'])
        
        return {
            'comparison_for': {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'required_capabilities': required_capabilities
            },
            'models': results,
            'cost_savings': self._calculate_savings(results)
        }
    
    def _calculate_savings(self, results: list) -> Dict[str, Any]:
        """Calcular ahorros potenciales"""
        
        if len(results) < 2:
            return {'savings_available': False}
        
        cheapest = results[0]
        most_expensive = results[-1]
        
        absolute_savings = most_expensive['total_cost_usd'] - cheapest['total_cost_usd']
        percentage_savings = (absolute_savings / most_expensive['total_cost_usd']) * 100
        
        return {
            'savings_available': True,
            'cheapest_model': cheapest['model'],
            'most_expensive_model': most_expensive['model'],
            'absolute_savings_usd': round(absolute_savings, 6),
            'percentage_savings': round(percentage_savings, 2),
            'monthly_savings_1k_requests': round(absolute_savings * 1000, 2)
        }
```

## 🔧 Estrategias de Optimización de Tokens

### 1. Optimizador de Prompts

```python
import re
import tiktoken
from typing import List, Dict, Any, Optional

class PromptOptimizer:
    """Optimizador de prompts para reducir tokens"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        
        # Diccionario de abreviaciones comunes
        self.abbreviations = {
            "information": "info",
            "application": "app",
            "development": "dev",
            "environment": "env",
            "configuration": "config",
            "documentation": "docs",
            "implementation": "impl",
            "optimization": "opt",
            "performance": "perf",
            "requirements": "reqs",
            "specifications": "specs"
        }
        
        # Palabras de relleno que se pueden eliminar
        self.filler_words = {
            "please", "kindly", "very", "really", "quite", "somewhat", 
            "rather", "fairly", "pretty", "just", "simply", "basically"
        }
    
    def count_tokens(self, text: str) -> int:
        """Contar tokens en el texto"""
        return len(self.encoding.encode(text))
    
    def optimize_prompt(self, prompt: str, aggressive: bool = False) -> Dict[str, Any]:
        """Optimizar prompt para reducir tokens"""
        
        original_tokens = self.count_tokens(prompt)
        optimized_prompt = prompt
        
        # 1. Eliminar espacios extra y saltos de línea innecesarios
        optimized_prompt = re.sub(r'\s+', ' ', optimized_prompt)
        optimized_prompt = re.sub(r'\n\s*\n', '\n', optimized_prompt)
        
        # 2. Eliminar palabras de relleno
        if aggressive:
            for filler in self.filler_words:
                pattern = r'\b' + re.escape(filler) + r'\b'
                optimized_prompt = re.sub(pattern, '', optimized_prompt, flags=re.IGNORECASE)
        
        # 3. Aplicar abreviaciones
        for full_word, abbrev in self.abbreviations.items():
            pattern = r'\b' + re.escape(full_word) + r'\b'
            optimized_prompt = re.sub(pattern, abbrev, optimized_prompt, flags=re.IGNORECASE)
        
        # 4. Optimizar estructura de listas
        optimized_prompt = self._optimize_lists(optimized_prompt)
        
        # 5. Comprimir instrucciones repetitivas
        optimized_prompt = self._compress_instructions(optimized_prompt)
        
        # 6. Limpiar espacios finales
        optimized_prompt = re.sub(r'\s+', ' ', optimized_prompt).strip()
        
        optimized_tokens = self.count_tokens(optimized_prompt)
        
        return {
            'original_prompt': prompt,
            'optimized_prompt': optimized_prompt,
            'original_tokens': original_tokens,
            'optimized_tokens': optimized_tokens,
            'tokens_saved': original_tokens - optimized_tokens,
            'reduction_percentage': round(((original_tokens - optimized_tokens) / original_tokens) * 100, 2),
            'estimated_cost_savings': self._estimate_cost_savings(original_tokens - optimized_tokens)
        }
    
    def _optimize_lists(self, text: str) -> str:
        """Optimizar formato de listas"""
        
        # Convertir listas verbosas a formato compacto
        # "1. First item\n2. Second item" -> "1) First item 2) Second item"
        text = re.sub(r'(\d+)\.\s+', r'\1) ', text)
        text = re.sub(r'\n(\d+\))', r' \1', text)
        
        # Optimizar bullets
        text = re.sub(r'•\s+', '• ', text)
        text = re.sub(r'\n•', ' •', text)
        
        return text
    
    def _compress_instructions(self, text: str) -> str:
        """Comprimir instrucciones repetitivas"""
        
        # Reemplazar frases comunes con versiones más cortas
        replacements = {
            "You are an AI assistant that": "You're an AI that",
            "Please make sure to": "Ensure you",
            "It is important that you": "You must",
            "In your response, please": "In response,",
            "Take into consideration": "Consider",
            "Make sure that you": "Ensure you",
            "Please note that": "Note:",
            "Keep in mind that": "Remember:"
        }
        
        for old_phrase, new_phrase in replacements.items():
            text = re.sub(re.escape(old_phrase), new_phrase, text, flags=re.IGNORECASE)
        
        return text
    
    def _estimate_cost_savings(self, tokens_saved: int) -> Dict[str, float]:
        """Estimar ahorros de costo basado en tokens ahorrados"""
        
        # Usar precios de GPT-4 como baseline
        input_price_per_1k = 0.03
        output_price_per_1k = 0.06
        
        # Asumir proporción 70% input, 30% output
        input_savings = (tokens_saved * 0.7 / 1000) * input_price_per_1k
        output_savings = (tokens_saved * 0.3 / 1000) * output_price_per_1k
        
        total_savings_per_request = input_savings + output_savings
        
        return {
            'per_request_usd': round(total_savings_per_request, 6),
            'per_1k_requests_usd': round(total_savings_per_request * 1000, 2),
            'per_month_100k_requests_usd': round(total_savings_per_request * 100000, 2)
        }
    
    def suggest_alternatives(self, prompt: str) -> List[Dict[str, Any]]:
        """Sugerir alternativas más eficientes"""
        
        suggestions = []
        
        # Detectar prompts muy largos
        if self.count_tokens(prompt) > 1000:
            suggestions.append({
                'type': 'length_reduction',
                'suggestion': 'Consider breaking this into multiple shorter prompts',
                'potential_savings': '20-40% token reduction'
            })
        
        # Detectar ejemplos excesivos
        example_count = len(re.findall(r'example\s*\d*:', prompt, re.IGNORECASE))
        if example_count > 3:
            suggestions.append({
                'type': 'example_optimization',
                'suggestion': f'Reduce from {example_count} examples to 2-3 most representative ones',
                'potential_savings': f'{(example_count - 2) * 50} tokens approximately'
            })
        
        # Detectar repetición de contexto
        sentences = prompt.split('.')
        if len(sentences) > 10:
            suggestions.append({
                'type': 'context_compression',
                'suggestion': 'Compress repeated context or use reference-based prompting',
                'potential_savings': '15-25% token reduction'
            })
        
        return suggestions
```

### 2. Sistema de Caching Inteligente

```python
import hashlib
import json
import time
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis
from dataclasses import dataclass, asdict

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int
    model_name: str
    prompt_hash: str
    cost_saved: float

class IntelligentLLMCache:
    """Sistema de caching inteligente para LLMs"""
    
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.cost_calculator = CostCalculator()
        
        # Configuración de TTL por tipo de prompt
        self.ttl_strategies = {
            'static_content': 86400 * 7,  # 1 semana
            'data_analysis': 3600,  # 1 hora
            'creative_writing': 1800,  # 30 minutos
            'code_generation': 7200,  # 2 horas
            'translation': 86400,  # 1 día
            'summarization': 3600,  # 1 hora
            'qa_factual': 86400 * 3,  # 3 días
            'conversation': 900  # 15 minutos
        }
    
    def _generate_cache_key(self, 
                          model_name: str, 
                          prompt: str, 
                          temperature: float = 0.7,
                          max_tokens: int = 150) -> str:
        """Generar clave de cache basada en parámetros"""
        
        # Normalizar prompt para mejor hit rate
        normalized_prompt = self._normalize_prompt(prompt)
        
        # Crear hash único
        cache_data = {
            'model': model_name,
            'prompt': normalized_prompt,
            'temperature': round(temperature, 2),
            'max_tokens': max_tokens
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"llm_cache:{hashlib.sha256(cache_string.encode()).hexdigest()}"
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalizar prompt para mejorar cache hit rate"""
        
        # Eliminar espacios extra
        normalized = re.sub(r'\s+', ' ', prompt.strip())
        
        # Normalizar fechas variables (mantener estructura)
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '[DATE]', normalized)
        normalized = re.sub(r'\d{1,2}:\d{2}', '[TIME]', normalized)
        
        # Normalizar IDs y números variables
        normalized = re.sub(r'\bid_\d+\b', '[ID]', normalized)
        normalized = re.sub(r'\b\d{6,}\b', '[NUMBER]', normalized)
        
        return normalized.lower()
    
    def _detect_prompt_type(self, prompt: str) -> str:
        """Detectar tipo de prompt para TTL estratégico"""
        
        prompt_lower = prompt.lower()
        
        # Patrones de detección
        patterns = {
            'translation': ['translate', 'translation', 'spanish to english', 'english to spanish'],
            'code_generation': ['write code', 'function', 'class', 'algorithm', 'programming'],
            'summarization': ['summarize', 'summary', 'brief', 'key points'],
            'qa_factual': ['what is', 'when did', 'where is', 'who is', 'define'],
            'creative_writing': ['write a story', 'poem', 'creative', 'narrative'],
            'data_analysis': ['analyze', 'data', 'statistics', 'trends', 'insights'],
            'conversation': ['chat', 'conversation', 'discuss', 'talk about']
        }
        
        for prompt_type, keywords in patterns.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return prompt_type
        
        return 'static_content'  # Default
    
    async def get(self, 
                 model_name: str, 
                 prompt: str, 
                 temperature: float = 0.7,
                 max_tokens: int = 150) -> Optional[Dict[str, Any]]:
        """Obtener respuesta del cache"""
        
        cache_key = self._generate_cache_key(model_name, prompt, temperature, max_tokens)
        
        try:
            cached_data = await self.redis.get(cache_key)
            
            if cached_data:
                cache_entry = json.loads(cached_data)
                
                # Actualizar estadísticas de acceso
                cache_entry['accessed_at'] = datetime.now().isoformat()
                cache_entry['access_count'] += 1
                
                # Extender TTL para contenido frecuentemente accedido
                if cache_entry['access_count'] > 10:
                    await self.redis.expire(cache_key, self.default_ttl * 2)
                
                # Guardar estadísticas actualizadas
                await self.redis.set(cache_key, json.dumps(cache_entry))
                
                return {
                    'cache_hit': True,
                    'response': cache_entry['value'],
                    'cached_at': cache_entry['created_at'],
                    'access_count': cache_entry['access_count'],
                    'cost_saved_usd': cache_entry['cost_saved']
                }
            
            return None
            
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, 
                 model_name: str, 
                 prompt: str, 
                 response: Any,
                 temperature: float = 0.7,
                 max_tokens: int = 150,
                 input_tokens: int = 0,
                 output_tokens: int = 0) -> bool:
        """Guardar respuesta en cache"""
        
        try:
            cache_key = self._generate_cache_key(model_name, prompt, temperature, max_tokens)
            prompt_type = self._detect_prompt_type(prompt)
            ttl = self.ttl_strategies.get(prompt_type, self.default_ttl)
            
            # Calcular costo ahorrado
            cost_info = self.cost_calculator.calculate_request_cost(
                model_name, input_tokens, output_tokens
            )
            
            cache_entry = {
                'key': cache_key,
                'value': response,
                'created_at': datetime.now().isoformat(),
                'accessed_at': datetime.now().isoformat(),
                'access_count': 0,
                'ttl_seconds': ttl,
                'model_name': model_name,
                'prompt_hash': hashlib.md5(prompt.encode()).hexdigest(),
                'cost_saved': cost_info['total_cost_usd'],
                'prompt_type': prompt_type
            }
            
            await self.redis.setex(cache_key, ttl, json.dumps(cache_entry))
            
            # Mantener índice de prompts por tipo para analytics
            await self.redis.sadd(f"prompt_types:{prompt_type}", cache_key)
            
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del cache"""
        
        try:
            # Obtener todas las claves de cache
            cache_keys = await self.redis.keys("llm_cache:*")
            
            total_entries = len(cache_keys)
            total_cost_saved = 0
            total_hits = 0
            type_distribution = {}
            
            for key in cache_keys:
                try:
                    cached_data = await self.redis.get(key)
                    if cached_data:
                        cache_entry = json.loads(cached_data)
                        total_cost_saved += cache_entry.get('cost_saved', 0)
                        total_hits += cache_entry.get('access_count', 0)
                        
                        prompt_type = cache_entry.get('prompt_type', 'unknown')
                        type_distribution[prompt_type] = type_distribution.get(prompt_type, 0) + 1
                        
                except:
                    continue
            
            # Calcular hit rate estimado
            hit_rate = (total_hits / max(total_entries, 1)) * 100
            
            return {
                'total_cached_entries': total_entries,
                'total_cache_hits': total_hits,
                'estimated_hit_rate_percent': round(hit_rate, 2),
                'total_cost_saved_usd': round(total_cost_saved, 4),
                'monthly_projected_savings_usd': round(total_cost_saved * 30, 2),
                'cache_distribution_by_type': type_distribution,
                'average_cost_saved_per_hit': round(total_cost_saved / max(total_hits, 1), 6)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidar cache por patrón"""
        
        try:
            keys = await self.redis.keys(f"llm_cache:*{pattern}*")
            if keys:
                return await self.redis.delete(*keys)
            return 0
            
        except Exception as e:
            print(f"Cache invalidation error: {e}")
            return 0
    
    async def cleanup_expired(self) -> Dict[str, Any]:
        """Limpiar entradas expiradas manualmente"""
        
        cleaned_count = 0
        total_count = 0
        
        try:
            cache_keys = await self.redis.keys("llm_cache:*")
            total_count = len(cache_keys)
            
            for key in cache_keys:
                ttl = await self.redis.ttl(key)
                if ttl == -2:  # Clave expirada
                    await self.redis.delete(key)
                    cleaned_count += 1
            
            return {
                'cleanup_completed': True,
                'total_entries_checked': total_count,
                'expired_entries_removed': cleaned_count,
                'remaining_entries': total_count - cleaned_count
            }
            
        except Exception as e:
            return {'cleanup_completed': False, 'error': str(e)}
```

### 3. Selector Inteligente de Modelos

```python
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class TaskRequirements:
    task_type: str
    complexity_level: str  # "simple", "medium", "complex"
    required_capabilities: List[str]
    max_acceptable_cost: float
    quality_threshold: float  # 0.0 - 1.0
    speed_requirement: str  # "fast", "medium", "slow"
    context_length_needed: int

class IntelligentModelSelector:
    """Selector inteligente de modelos basado en requerimientos"""
    
    def __init__(self):
        self.cost_calculator = CostCalculator()
        
        # Matriz de calidad por tarea (scores 0.0 - 1.0)
        self.quality_matrix = {
            "gpt-4-turbo": {
                "reasoning": 0.95,
                "coding": 0.92,
                "creative_writing": 0.88,
                "analysis": 0.94,
                "translation": 0.85,
                "summarization": 0.89,
                "qa_factual": 0.93,
                "conversation": 0.87
            },
            "gpt-4": {
                "reasoning": 0.93,
                "coding": 0.90,
                "creative_writing": 0.90,
                "analysis": 0.92,
                "translation": 0.83,
                "summarization": 0.87,
                "qa_factual": 0.91,
                "conversation": 0.85
            },
            "gpt-3.5-turbo": {
                "reasoning": 0.75,
                "coding": 0.78,
                "creative_writing": 0.82,
                "analysis": 0.76,
                "translation": 0.80,
                "summarization": 0.83,
                "qa_factual": 0.81,
                "conversation": 0.88
            },
            "claude-3-opus": {
                "reasoning": 0.96,
                "coding": 0.89,
                "creative_writing": 0.94,
                "analysis": 0.95,
                "translation": 0.87,
                "summarization": 0.92,
                "qa_factual": 0.94,
                "conversation": 0.91
            },
            "claude-3-sonnet": {
                "reasoning": 0.88,
                "coding": 0.85,
                "creative_writing": 0.89,
                "analysis": 0.87,
                "translation": 0.84,
                "summarization": 0.88,
                "qa_factual": 0.86,
                "conversation": 0.85
            },
            "claude-3-haiku": {
                "reasoning": 0.72,
                "coding": 0.74,
                "creative_writing": 0.76,
                "analysis": 0.70,
                "translation": 0.78,
                "summarization": 0.79,
                "qa_factual": 0.75,
                "conversation": 0.81
            }
        }
        
        # Velocidad relativa (requests per minute)
        self.speed_ratings = {
            "gpt-3.5-turbo": 100,
            "claude-3-haiku": 95,
            "claude-3-sonnet": 60,
            "gpt-4-turbo": 40,
            "gpt-4": 30,
            "claude-3-opus": 20
        }
    
    def select_optimal_model(self, 
                           requirements: TaskRequirements,
                           estimated_input_tokens: int = 1000,
                           estimated_output_tokens: int = 500) -> Dict[str, Any]:
        """Seleccionar el modelo óptimo basado en requerimientos"""
        
        candidates = []
        
        for model_name, pricing in self.cost_calculator.pricing_data.items():
            # Filtrar por capacidades requeridas
            if requirements.required_capabilities:
                if not all(cap in pricing.capabilities for cap in requirements.required_capabilities):
                    continue
            
            # Filtrar por context window
            if pricing.context_window < requirements.context_length_needed:
                continue
            
            # Calcular costo
            cost_info = self.cost_calculator.calculate_request_cost(
                model_name, estimated_input_tokens, estimated_output_tokens
            )
            
            # Filtrar por presupuesto
            if cost_info['total_cost_usd'] > requirements.max_acceptable_cost:
                continue
            
            # Obtener score de calidad
            quality_score = self.quality_matrix.get(model_name, {}).get(
                requirements.task_type, 0.5
            )
            
            # Filtrar por calidad mínima
            if quality_score < requirements.quality_threshold:
                continue
            
            # Calcular score de velocidad
            speed_score = self._calculate_speed_score(model_name, requirements.speed_requirement)
            
            # Calcular score general
            overall_score = self._calculate_overall_score(
                quality_score, cost_info['total_cost_usd'], speed_score, requirements
            )
            
            candidates.append({
                'model_name': model_name,
                'provider': pricing.provider.value,
                'quality_score': quality_score,
                'cost_usd': cost_info['total_cost_usd'],
                'speed_score': speed_score,
                'overall_score': overall_score,
                'cost_info': cost_info,
                'meets_requirements': True,
                'performance_tier': pricing.performance_tier
            })
        
        # Ordenar por score general
        candidates.sort(key=lambda x: x['overall_score'], reverse=True)
        
        if not candidates:
            return {
                'success': False,
                'message': 'No models meet the specified requirements',
                'suggestions': self._generate_requirement_suggestions(requirements)
            }
        
        recommended = candidates[0]
        alternatives = candidates[1:3]  # Top 3 alternativas
        
        return {
            'success': True,
            'recommended_model': recommended,
            'alternatives': alternatives,
            'selection_rationale': self._generate_rationale(recommended, requirements),
            'cost_comparison': self._generate_cost_comparison(candidates),
            'total_candidates_evaluated': len(candidates)
        }
    
    def _calculate_speed_score(self, model_name: str, speed_requirement: str) -> float:
        """Calcular score de velocidad"""
        
        base_speed = self.speed_ratings.get(model_name, 50)
        max_speed = max(self.speed_ratings.values())
        
        # Normalizar a 0.0-1.0
        normalized_speed = base_speed / max_speed
        
        # Aplicar peso según requerimiento
        if speed_requirement == "fast":
            return normalized_speed
        elif speed_requirement == "medium":
            return min(normalized_speed + 0.2, 1.0)  # Bonus moderado
        else:  # slow
            return min(normalized_speed + 0.4, 1.0)  # Bonus alto
    
    def _calculate_overall_score(self, 
                               quality_score: float, 
                               cost: float, 
                               speed_score: float,
                               requirements: TaskRequirements) -> float:
        """Calcular score general ponderado"""
        
        # Normalizar costo (invertir para que menor costo = mayor score)
        max_cost = requirements.max_acceptable_cost
        cost_score = 1.0 - (cost / max_cost)
        
        # Pesos según complejidad de la tarea
        if requirements.complexity_level == "simple":
            weights = {'cost': 0.5, 'speed': 0.3, 'quality': 0.2}
        elif requirements.complexity_level == "medium":
            weights = {'cost': 0.3, 'speed': 0.2, 'quality': 0.5}
        else:  # complex
            weights = {'cost': 0.2, 'speed': 0.1, 'quality': 0.7}
        
        overall_score = (
            quality_score * weights['quality'] +
            cost_score * weights['cost'] +
            speed_score * weights['speed']
        )
        
        return round(overall_score, 3)
    
    def _generate_rationale(self, 
                          recommended: Dict[str, Any], 
                          requirements: TaskRequirements) -> List[str]:
        """Generar explicación de la selección"""
        
        rationale = []
        
        rationale.append(f"Selected {recommended['model_name']} for {requirements.task_type} task")
        
        if recommended['quality_score'] >= 0.9:
            rationale.append("High quality score ensures excellent output")
        elif recommended['quality_score'] >= 0.8:
            rationale.append("Good quality score meets requirements effectively")
        
        if recommended['cost_usd'] <= requirements.max_acceptable_cost * 0.5:
            rationale.append("Cost-efficient choice, well below budget")
        elif recommended['cost_usd'] <= requirements.max_acceptable_cost * 0.8:
            rationale.append("Reasonable cost within budget constraints")
        
        if recommended['speed_score'] >= 0.8:
            rationale.append("Fast response time meets speed requirements")
        
        return rationale
    
    def _generate_cost_comparison(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar comparación de costos"""
        
        if len(candidates) < 2:
            return {'comparison_available': False}
        
        costs = [c['cost_usd'] for c in candidates]
        cheapest_cost = min(costs)
        most_expensive_cost = max(costs)
        
        recommended_cost = candidates[0]['cost_usd']
        
        return {
            'comparison_available': True,
            'recommended_model_cost': recommended_cost,
            'cheapest_option_cost': cheapest_cost,
            'most_expensive_cost': most_expensive_cost,
            'cost_vs_cheapest_percent': round(((recommended_cost - cheapest_cost) / cheapest_cost) * 100, 1) if cheapest_cost > 0 else 0,
            'savings_vs_most_expensive': round(most_expensive_cost - recommended_cost, 6)
        }
    
    def _generate_requirement_suggestions(self, requirements: TaskRequirements) -> List[str]:
        """Generar sugerencias para ajustar requerimientos"""
        
        suggestions = []
        
        if requirements.max_acceptable_cost < 0.001:
            suggestions.append("Consider increasing budget - minimum viable cost is ~$0.001 per request")
        
        if requirements.quality_threshold > 0.95:
            suggestions.append("Quality threshold very high - consider 0.85-0.9 for good results")
        
        if requirements.context_length_needed > 100000:
            suggestions.append("Very large context window - consider breaking into smaller chunks")
        
        return suggestions
    
    def batch_optimize(self, 
                      tasks: List[Tuple[TaskRequirements, int, int]]) -> Dict[str, Any]:
        """Optimizar selección para múltiples tareas"""
        
        optimizations = []
        total_cost_original = 0
        total_cost_optimized = 0
        
        for requirements, input_tokens, output_tokens in tasks:
            # Selección individual óptima
            result = self.select_optimal_model(requirements, input_tokens, output_tokens)
            
            if result['success']:
                optimizations.append({
                    'task_type': requirements.task_type,
                    'recommended_model': result['recommended_model']['model_name'],
                    'cost_usd': result['recommended_model']['cost_usd'],
                    'quality_score': result['recommended_model']['quality_score']
                })
                
                total_cost_optimized += result['recommended_model']['cost_usd']
                
                # Costo si usáramos siempre GPT-4 (baseline)
                baseline_cost = self.cost_calculator.calculate_request_cost(
                    "gpt-4", input_tokens, output_tokens
                )['total_cost_usd']
                total_cost_original += baseline_cost
        
        savings = total_cost_original - total_cost_optimized
        savings_percentage = (savings / total_cost_original * 100) if total_cost_original > 0 else 0
        
        return {
            'batch_optimization': True,
            'total_tasks': len(tasks),
            'optimizations': optimizations,
            'cost_analysis': {
                'baseline_cost_usd': round(total_cost_original, 4),
                'optimized_cost_usd': round(total_cost_optimized, 4),
                'total_savings_usd': round(savings, 4),
                'savings_percentage': round(savings_percentage, 2)
            },
            'model_distribution': self._analyze_model_distribution(optimizations)
        }
    
    def _analyze_model_distribution(self, optimizations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analizar distribución de modelos seleccionados"""
        
        distribution = {}
        
        for opt in optimizations:
            model = opt['recommended_model']
            distribution[model] = distribution.get(model, 0) + 1
        
        return distribution
```

## 📊 Monitoreo de Costos en Tiempo Real

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class CostMonitor:
    """Monitor de costos en tiempo real"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cost_calculator = CostCalculator()
        
        # Configuración de alertas
        self.alert_thresholds = {
            'daily_budget': 100.0,
            'hourly_rate': 10.0,
            'per_request_max': 1.0,
            'monthly_budget': 2000.0
        }
    
    async def track_request(self, 
                          model_name: str,
                          input_tokens: int,
                          output_tokens: int,
                          user_id: str = None,
                          request_type: str = "api") -> Dict[str, Any]:
        """Trackear costo de una request"""
        
        timestamp = datetime.now()
        cost_info = self.cost_calculator.calculate_request_cost(
            model_name, input_tokens, output_tokens
        )
        
        # Crear registro de costo
        cost_record = {
            'timestamp': timestamp.isoformat(),
            'model_name': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_usd': cost_info['total_cost_usd'],
            'user_id': user_id,
            'request_type': request_type
        }
        
        # Guardar en Redis con diferentes keys para agregación
        await self._store_cost_record(cost_record, timestamp)
        
        # Verificar alertas
        alerts = await self._check_alerts(cost_info['total_cost_usd'], timestamp)
        
        return {
            'cost_tracked': True,
            'cost_record': cost_record,
            'running_totals': await self._get_running_totals(timestamp),
            'alerts': alerts
        }
    
    async def _store_cost_record(self, record: Dict[str, Any], timestamp: datetime):
        """Almacenar registro de costo en múltiples agregaciones"""
        
        # Key por minuto para granularidad alta
        minute_key = f"costs:minute:{timestamp.strftime('%Y%m%d:%H%M')}"
        
        # Key por hora
        hour_key = f"costs:hour:{timestamp.strftime('%Y%m%d:%H')}"
        
        # Key por día
        day_key = f"costs:day:{timestamp.strftime('%Y%m%d')}"
        
        # Key por mes
        month_key = f"costs:month:{timestamp.strftime('%Y%m')}"
        
        # Almacenar registro completo
        record_key = f"cost_record:{timestamp.timestamp()}"
        await self.redis.setex(record_key, 86400 * 7, json.dumps(record))  # 7 días TTL
        
        # Agregar a listas de agregación
        cost_value = record['cost_usd']
        
        # Incrementar contadores y sumas
        pipe = self.redis.pipeline()
        
        # Por minuto
        pipe.hincrbyfloat(minute_key, 'total_cost', cost_value)
        pipe.hincrby(minute_key, 'request_count', 1)
        pipe.hincrby(minute_key, 'total_tokens', record['total_tokens'])
        pipe.expire(minute_key, 86400)  # 1 día TTL
        
        # Por hora
        pipe.hincrbyfloat(hour_key, 'total_cost', cost_value)
        pipe.hincrby(hour_key, 'request_count', 1)
        pipe.hincrby(hour_key, 'total_tokens', record['total_tokens'])
        pipe.expire(hour_key, 86400 * 7)  # 7 días TTL
        
        # Por día
        pipe.hincrbyfloat(day_key, 'total_cost', cost_value)
        pipe.hincrby(day_key, 'request_count', 1)
        pipe.hincrby(day_key, 'total_tokens', record['total_tokens'])
        pipe.expire(day_key, 86400 * 30)  # 30 días TTL
        
        # Por mes
        pipe.hincrbyfloat(month_key, 'total_cost', cost_value)
        pipe.hincrby(month_key, 'request_count', 1)
        pipe.hincrby(month_key, 'total_tokens', record['total_tokens'])
        pipe.expire(month_key, 86400 * 365)  # 1 año TTL
        
        await pipe.execute()
    
    async def _get_running_totals(self, timestamp: datetime) -> Dict[str, Any]:
        """Obtener totales acumulados"""
        
        day_key = f"costs:day:{timestamp.strftime('%Y%m%d')}"
        month_key = f"costs:month:{timestamp.strftime('%Y%m')}"
        
        day_data = await self.redis.hgetall(day_key)
        month_data = await self.redis.hgetall(month_key)
        
        return {
            'today': {
                'total_cost_usd': float(day_data.get(b'total_cost', 0)),
                'request_count': int(day_data.get(b'request_count', 0)),
                'total_tokens': int(day_data.get(b'total_tokens', 0))
            },
            'this_month': {
                'total_cost_usd': float(month_data.get(b'total_cost', 0)),
                'request_count': int(month_data.get(b'request_count', 0)),
                'total_tokens': int(month_data.get(b'total_tokens', 0))
            }
        }
    
    async def _check_alerts(self, current_cost: float, timestamp: datetime) -> List[Dict[str, Any]]:
        """Verificar si se deben disparar alertas"""
        
        alerts = []
        
        # Alert por costo individual alto
        if current_cost > self.alert_thresholds['per_request_max']:
            alerts.append({
                'type': 'high_individual_cost',
                'message': f'Single request cost ${current_cost:.4f} exceeds threshold',
                'severity': 'warning',
                'threshold': self.alert_thresholds['per_request_max']
            })
        
        # Alert por presupuesto diario
        totals = await self._get_running_totals(timestamp)
        daily_cost = totals['today']['total_cost_usd']
        
        if daily_cost > self.alert_thresholds['daily_budget'] * 0.8:
            alerts.append({
                'type': 'daily_budget_warning',
                'message': f'Daily cost ${daily_cost:.2f} approaching limit',
                'severity': 'warning' if daily_cost < self.alert_thresholds['daily_budget'] else 'critical',
                'current': daily_cost,
                'threshold': self.alert_thresholds['daily_budget']
            })
        
        # Alert por presupuesto mensual
        monthly_cost = totals['this_month']['total_cost_usd']
        
        if monthly_cost > self.alert_thresholds['monthly_budget'] * 0.8:
            alerts.append({
                'type': 'monthly_budget_warning',
                'message': f'Monthly cost ${monthly_cost:.2f} approaching limit',
                'severity': 'warning' if monthly_cost < self.alert_thresholds['monthly_budget'] else 'critical',
                'current': monthly_cost,
                'threshold': self.alert_thresholds['monthly_budget']
            })
        
        return alerts
    
    async def get_cost_analytics(self, 
                               period: str = "24h",
                               granularity: str = "hour") -> Dict[str, Any]:
        """Obtener analytics de costos"""
        
        now = datetime.now()
        
        if period == "24h":
            start_time = now - timedelta(hours=24)
            time_format = '%Y%m%d:%H'
            key_prefix = "costs:hour"
        elif period == "7d":
            start_time = now - timedelta(days=7)
            time_format = '%Y%m%d'
            key_prefix = "costs:day"
        elif period == "30d":
            start_time = now - timedelta(days=30)
            time_format = '%Y%m%d'
            key_prefix = "costs:day"
        else:
            raise ValueError("Invalid period. Use '24h', '7d', or '30d'")
        
        # Generar keys para el periodo
        current_time = start_time
        time_series = []
        total_cost = 0
        total_requests = 0
        total_tokens = 0
        
        while current_time <= now:
            key = f"{key_prefix}:{current_time.strftime(time_format)}"
            data = await self.redis.hgetall(key)
            
            cost = float(data.get(b'total_cost', 0))
            requests = int(data.get(b'request_count', 0))
            tokens = int(data.get(b'total_tokens', 0))
            
            time_series.append({
                'timestamp': current_time.isoformat(),
                'cost_usd': cost,
                'request_count': requests,
                'token_count': tokens,
                'avg_cost_per_request': cost / max(requests, 1)
            })
            
            total_cost += cost
            total_requests += requests
            total_tokens += tokens
            
            # Incrementar tiempo
            if granularity == "hour":
                current_time += timedelta(hours=1)
            else:
                current_time += timedelta(days=1)
        
        return {
            'period': period,
            'granularity': granularity,
            'time_series': time_series,
            'summary': {
                'total_cost_usd': round(total_cost, 4),
                'total_requests': total_requests,
                'total_tokens': total_tokens,
                'avg_cost_per_request': round(total_cost / max(total_requests, 1), 6),
                'avg_cost_per_token': round(total_cost / max(total_tokens, 1), 8)
            },
            'projected_monthly_cost': round(total_cost * (30 / max((now - start_time).days, 1)), 2)
        }
```

## ✅ Mejores Prácticas de Optimización

### 1. **Estratificación de Modelos**
- **Tareas simples**: Claude-3 Haiku, GPT-3.5 Turbo
- **Tareas complejas**: GPT-4 Turbo, Claude-3 Opus
- **Casos específicos**: Modelos especializados

### 2. **Optimización de Prompts**
- Eliminar palabras innecesarias
- Usar abreviaciones consistentes
- Estructurar información de forma compacta
- Reutilizar templates optimizados

### 3. **Caching Inteligente**
- TTL dinámico por tipo de contenido
- Normalización de prompts para mejor hit rate
- Invalidación selectiva por patrones

### 4. **Monitoreo Continuo**
- Alertas por presupuesto
- Analytics en tiempo real
- Proyecciones de costos

## 🎯 Próximo Paso

En el **Laboratorio 5** implementaremos un sistema completo de optimización de costos con caching, selección automática de modelos y monitoreo en tiempo real.

## 📖 Recursos Adicionales

- [OpenAI Pricing](https://openai.com/pricing)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [Token Optimization Guide](https://help.openai.com/en/articles/4936856)
- [LLM Cost Analysis Tools](https://github.com/microsoft/llmops-template)
