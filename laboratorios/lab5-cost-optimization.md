# 🧪 Lab 5: Sistema de Optimización de Costos y Caching

## 🎯 Objetivos del Laboratorio

Al completar este laboratorio podrás:
- Implementar un sistema completo de caching para LLMs
- Configurar selección automática de modelos por costo/calidad
- Implementar monitoreo de costos en tiempo real
- Crear dashboards de analytics de uso y gastos
- Optimizar prompts automáticamente para reducir tokens

## 📋 Prerrequisitos

- Docker y Docker Compose instalados
- Redis para caching
- PostgreSQL para analytics
- Prometheus y Grafana para monitoreo
- API keys de OpenAI y Anthropic

## 🏗️ Arquitectura del Sistema

```mermaid
graph TB
    A[Client Request] --> B[Cost Optimizer API]
    B --> C[Prompt Optimizer]
    B --> D[Cache Layer]
    B --> E[Model Selector]
    
    C --> F[Token Counter]
    D --> G[Redis Cache]
    E --> H[Cost Calculator]
    
    H --> I[Model Routing]
    I --> J[OpenAI API]
    I --> K[Anthropic API]
    
    B --> L[Cost Monitor]
    L --> M[PostgreSQL]
    L --> N[Prometheus Metrics]
    
    N --> O[Grafana Dashboard]
    M --> O
```

## 🚀 Configuración del Entorno

### 1. Estructura del Proyecto

```bash
cost-optimization-lab/
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── prompt_optimizer.py
│   │   ├── cache_manager.py
│   │   ├── model_selector.py
│   │   └── cost_monitor.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── models.py
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── openai_client.py
│   │   └── anthropic_client.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
├── tests/
│   ├── test_optimization.py
│   ├── test_caching.py
│   └── load_test.py
├── dashboards/
│   ├── cost_analytics.json
│   └── performance_metrics.json
├── config/
│   ├── models_config.yaml
│   └── optimization_rules.yaml
└── sql/
    └── init_tables.sql
```

### 2. Docker Compose Setup

```yaml
version: '3.8'

services:
  cost-optimizer-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://admin:secret@postgres:5432/cost_analytics
      - PROMETHEUS_GATEWAY=http://prometheus-pushgateway:9091
    volumes:
      - ./src:/app/src
      - ./config:/app/config
    depends_on:
      - redis
      - postgres
      - prometheus-pushgateway

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: cost_analytics
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secret
    ports:
      - "5433:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init_tables.sql:/docker-entrypoint-initdb.d/init.sql

  prometheus-pushgateway:
    image: prom/pushgateway
    ports:
      - "9091:9091"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

### 3. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar tiktoken para conteo de tokens
RUN pip install tiktoken

COPY . .

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
```

### 4. Requirements.txt

```txt
# Core API
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# LLM Clients
openai==1.3.7
anthropic==0.7.7

# Token counting
tiktoken==0.5.1

# Caching & Database
redis==5.0.1
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23

# Monitoring & Metrics
prometheus-client==0.19.0
structlog==23.2.0

# Optimization
numpy==1.24.3
scipy==1.11.4

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
```

## 💻 Implementación

### 1. Optimizador de Prompts Avanzado

```python
# src/optimization/prompt_optimizer.py
import re
import tiktoken
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter

@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    reduction_percentage: float
    estimated_savings_usd: float
    optimizations_applied: List[str]

class AdvancedPromptOptimizer:
    """Optimizador avanzado de prompts con ML y análisis semántico"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)
        
        # Configuración de optimización
        self.optimization_config = {
            'aggressive_mode': False,
            'preserve_examples': True,
            'min_compression_ratio': 0.1,  # Mínimo 10% de reducción
            'semantic_preservation': True
        }
        
        # Patrones de optimización
        self._initialize_optimization_patterns()
        
        # Cache de optimizaciones
        self.optimization_cache = {}
    
    def _initialize_optimization_patterns(self):
        """Inicializar patrones de optimización"""
        
        # Replacements de frases comunes
        self.phrase_replacements = {
            "you are an ai assistant that": "you're an ai that",
            "please make sure to": "ensure you",
            "it is important that you": "you must",
            "in your response, please": "in response,",
            "take into consideration": "consider",
            "keep in mind that": "remember:",
            "please note that": "note:",
            "make sure that you": "ensure you",
            "i would like you to": "please",
            "could you please": "please",
            "would you mind": "please",
            "if you could": "please"
        }
        
        # Patrones de compresión de listas
        self.list_patterns = [
            (r'(\d+)\.\s+([^\n]+)\n', r'\1) \2 '),  # "1. item" -> "1) item"
            (r'[-•]\s+([^\n]+)\n', r'• \1 '),        # "- item" -> "• item"
            (r'\n\s*\n', '\n'),                      # Múltiples líneas vacías
        ]
        
        # Palabras de relleno que se pueden eliminar en modo agresivo
        self.filler_words = {
            'very', 'really', 'quite', 'rather', 'fairly', 'pretty', 
            'just', 'simply', 'basically', 'actually', 'essentially',
            'literally', 'definitely', 'absolutely', 'completely',
            'totally', 'entirely', 'extremely', 'incredibly'
        }
        
        # Abreviaciones técnicas
        self.technical_abbreviations = {
            'application': 'app',
            'configuration': 'config',
            'development': 'dev',
            'environment': 'env',
            'information': 'info',
            'documentation': 'docs',
            'implementation': 'impl',
            'optimization': 'opt',
            'performance': 'perf',
            'requirements': 'reqs',
            'specifications': 'specs',
            'administration': 'admin',
            'authentication': 'auth',
            'authorization': 'authz',
            'database': 'db',
            'repository': 'repo',
            'directory': 'dir',
            'parameter': 'param',
            'parameters': 'params',
            'variable': 'var',
            'variables': 'vars'
        }
    
    async def optimize_prompt(self, 
                            prompt: str, 
                            optimization_level: str = "standard",
                            preserve_structure: bool = True) -> OptimizationResult:
        """Optimizar prompt con diferentes niveles de agresividad"""
        
        # Verificar cache
        cache_key = f"{hash(prompt)}_{optimization_level}_{preserve_structure}"
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        original_tokens = self.count_tokens(prompt)
        optimized_prompt = prompt
        optimizations_applied = []
        
        # Configurar nivel de optimización
        if optimization_level == "conservative":
            self.optimization_config.update({
                'aggressive_mode': False,
                'preserve_examples': True,
                'min_compression_ratio': 0.05
            })
        elif optimization_level == "aggressive":
            self.optimization_config.update({
                'aggressive_mode': True,
                'preserve_examples': False,
                'min_compression_ratio': 0.2
            })
        else:  # standard
            self.optimization_config.update({
                'aggressive_mode': False,
                'preserve_examples': True,
                'min_compression_ratio': 0.1
            })
        
        # 1. Normalización básica
        optimized_prompt, applied = self._normalize_whitespace(optimized_prompt)
        optimizations_applied.extend(applied)
        
        # 2. Compresión de frases comunes
        optimized_prompt, applied = self._compress_common_phrases(optimized_prompt)
        optimizations_applied.extend(applied)
        
        # 3. Optimización de listas y estructura
        if not preserve_structure:
            optimized_prompt, applied = self._optimize_structure(optimized_prompt)
            optimizations_applied.extend(applied)
        
        # 4. Abreviaciones técnicas
        optimized_prompt, applied = self._apply_abbreviations(optimized_prompt)
        optimizations_applied.extend(applied)
        
        # 5. Eliminación de palabras de relleno (modo agresivo)
        if self.optimization_config['aggressive_mode']:
            optimized_prompt, applied = self._remove_filler_words(optimized_prompt)
            optimizations_applied.extend(applied)
        
        # 6. Compresión de ejemplos (si permitido)
        if not self.optimization_config['preserve_examples']:
            optimized_prompt, applied = self._compress_examples(optimized_prompt)
            optimizations_applied.extend(applied)
        
        # 7. Optimización final
        optimized_prompt, applied = self._final_cleanup(optimized_prompt)
        optimizations_applied.extend(applied)
        
        # Calcular métricas
        optimized_tokens = self.count_tokens(optimized_prompt)
        reduction_percentage = ((original_tokens - optimized_tokens) / original_tokens) * 100
        
        # Verificar si cumple con criterios mínimos
        if reduction_percentage < self.optimization_config['min_compression_ratio'] * 100:
            # Si no hay suficiente mejora, devolver original
            result = OptimizationResult(
                original_prompt=prompt,
                optimized_prompt=prompt,
                original_tokens=original_tokens,
                optimized_tokens=original_tokens,
                reduction_percentage=0.0,
                estimated_savings_usd=0.0,
                optimizations_applied=["no_optimization_needed"]
            )
        else:
            # Calcular ahorro estimado
            estimated_savings = self._calculate_cost_savings(original_tokens - optimized_tokens)
            
            result = OptimizationResult(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                original_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
                reduction_percentage=round(reduction_percentage, 2),
                estimated_savings_usd=estimated_savings,
                optimizations_applied=list(set(optimizations_applied))
            )
        
        # Guardar en cache
        self.optimization_cache[cache_key] = result
        
        return result
    
    def _normalize_whitespace(self, text: str) -> Tuple[str, List[str]]:
        """Normalizar espacios en blanco"""
        
        applied = []
        
        # Eliminar espacios extra
        original_length = len(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 líneas vacías
        text = text.strip()
        
        if len(text) < original_length:
            applied.append("whitespace_normalization")
        
        return text, applied
    
    def _compress_common_phrases(self, text: str) -> Tuple[str, List[str]]:
        """Comprimir frases comunes"""
        
        applied = []
        original_text = text
        
        for original_phrase, compressed_phrase in self.phrase_replacements.items():
            pattern = re.compile(re.escape(original_phrase), re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(compressed_phrase, text)
                applied.append(f"phrase_compression_{original_phrase.replace(' ', '_')}")
        
        return text, applied
    
    def _optimize_structure(self, text: str) -> Tuple[str, List[str]]:
        """Optimizar estructura de listas y formato"""
        
        applied = []
        
        for pattern, replacement in self.list_patterns:
            original_text = text
            text = re.sub(pattern, replacement, text)
            if text != original_text:
                applied.append("structure_optimization")
        
        return text, applied
    
    def _apply_abbreviations(self, text: str) -> Tuple[str, List[str]]:
        """Aplicar abreviaciones técnicas"""
        
        applied = []
        
        for full_word, abbreviation in self.technical_abbreviations.items():
            pattern = r'\b' + re.escape(full_word) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, abbreviation, text, flags=re.IGNORECASE)
                applied.append(f"abbreviation_{full_word}")
        
        return text, applied
    
    def _remove_filler_words(self, text: str) -> Tuple[str, List[str]]:
        """Eliminar palabras de relleno en modo agresivo"""
        
        applied = []
        
        for filler_word in self.filler_words:
            pattern = r'\b' + re.escape(filler_word) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                applied.append(f"filler_removal_{filler_word}")
        
        # Limpiar espacios dobles resultantes
        text = re.sub(r'\s+', ' ', text)
        
        return text, applied
    
    def _compress_examples(self, text: str) -> Tuple[str, List[str]]:
        """Comprimir ejemplos manteniendo información esencial"""
        
        applied = []
        
        # Detectar secciones de ejemplos
        example_patterns = [
            r'(?i)examples?:\s*\n(.*?)(?=\n[A-Z]|\n\n|$)',
            r'(?i)for example:\s*(.*?)(?=\n[A-Z]|\n\n|$)',
            r'(?i)e\.g\.[:,]?\s*(.*?)(?=\n[A-Z]|\n\n|$)'
        ]
        
        for pattern in example_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                example_section = match.group(1)
                
                # Comprimir ejemplos múltiples a los más representativos
                if example_section.count('\n') > 3:  # Más de 3 líneas
                    lines = example_section.strip().split('\n')
                    # Mantener primer y último ejemplo
                    compressed = f"{lines[0]}\n...\n{lines[-1]}"
                    text = text.replace(example_section, compressed)
                    applied.append("example_compression")
        
        return text, applied
    
    def _final_cleanup(self, text: str) -> Tuple[str, List[str]]:
        """Limpieza final del texto"""
        
        applied = []
        
        # Eliminar espacios antes de puntuación
        original_text = text
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Comprimir múltiples espacios
        text = re.sub(r'\s+', ' ', text)
        
        # Eliminar líneas vacías al final
        text = text.strip()
        
        if text != original_text:
            applied.append("final_cleanup")
        
        return text, applied
    
    def count_tokens(self, text: str) -> int:
        """Contar tokens usando tiktoken"""
        return len(self.encoding.encode(text))
    
    def _calculate_cost_savings(self, tokens_saved: int) -> float:
        """Calcular ahorro estimado en USD"""
        
        # Usar precios de GPT-4 como baseline
        input_price_per_1k = 0.03
        
        # Asumir que los tokens ahorrados son principalmente de input
        savings = (tokens_saved / 1000) * input_price_per_1k
        
        return round(savings, 6)
    
    async def batch_optimize(self, prompts: List[str], optimization_level: str = "standard") -> Dict[str, Any]:
        """Optimizar múltiples prompts en lote"""
        
        tasks = []
        for prompt in prompts:
            tasks.append(self.optimize_prompt(prompt, optimization_level))
        
        results = await asyncio.gather(*tasks)
        
        # Calcular estadísticas agregadas
        total_original_tokens = sum(r.original_tokens for r in results)
        total_optimized_tokens = sum(r.optimized_tokens for r in results)
        total_savings = sum(r.estimated_savings_usd for r in results)
        
        optimization_frequency = Counter()
        for result in results:
            optimization_frequency.update(result.optimizations_applied)
        
        return {
            'total_prompts': len(prompts),
            'successful_optimizations': len([r for r in results if r.reduction_percentage > 0]),
            'total_tokens_saved': total_original_tokens - total_optimized_tokens,
            'total_cost_savings_usd': round(total_savings, 4),
            'average_reduction_percentage': round(
                sum(r.reduction_percentage for r in results) / len(results), 2
            ),
            'optimization_frequency': dict(optimization_frequency.most_common(10)),
            'individual_results': results
        }
    
    def analyze_prompt_efficiency(self, prompt: str) -> Dict[str, Any]:
        """Analizar eficiencia del prompt sin optimizar"""
        
        tokens = self.count_tokens(prompt)
        words = len(prompt.split())
        chars = len(prompt)
        
        # Análisis de estructura
        lines = prompt.split('\n')
        empty_lines = sum(1 for line in lines if not line.strip())
        
        # Detectar patrones ineficientes
        inefficiencies = []
        
        # Espacios múltiples
        if re.search(r'\s{3,}', prompt):
            inefficiencies.append("multiple_spaces")
        
        # Frases redundantes
        for phrase in self.phrase_replacements.keys():
            if phrase.lower() in prompt.lower():
                inefficiencies.append(f"redundant_phrase_{phrase.replace(' ', '_')}")
        
        # Palabras de relleno excesivas
        filler_count = sum(1 for word in prompt.lower().split() if word in self.filler_words)
        if filler_count > words * 0.1:  # Más del 10% son filler words
            inefficiencies.append("excessive_filler_words")
        
        # Ejemplos largos
        if 'example' in prompt.lower() and prompt.count('\n') > 10:
            inefficiencies.append("verbose_examples")
        
        return {
            'token_count': tokens,
            'word_count': words,
            'character_count': chars,
            'line_count': len(lines),
            'empty_lines': empty_lines,
            'tokens_per_word': round(tokens / max(words, 1), 2),
            'efficiency_score': self._calculate_efficiency_score(tokens, words, chars, inefficiencies),
            'detected_inefficiencies': inefficiencies,
            'optimization_potential': len(inefficiencies) > 0
        }
    
    def _calculate_efficiency_score(self, tokens: int, words: int, chars: int, inefficiencies: List[str]) -> float:
        """Calcular score de eficiencia (0.0 - 1.0, donde 1.0 es más eficiente)"""
        
        # Score base basado en ratio tokens/words (ideal ~1.3)
        token_word_ratio = tokens / max(words, 1)
        ratio_score = max(0, 1 - abs(token_word_ratio - 1.3) / 2)
        
        # Penalizar ineficiencias
        inefficiency_penalty = len(inefficiencies) * 0.1
        
        # Score final
        efficiency_score = max(0, ratio_score - inefficiency_penalty)
        
        return round(efficiency_score, 2)
```

### 2. Cache Manager con Estrategias Avanzadas

```python
# src/optimization/cache_manager.py
import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import redis.asyncio as redis
import pickle
import numpy as np

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
    cache_strategy: str
    hit_prediction_score: float = 0.0

class SmartCacheManager:
    """Cache manager inteligente con ML para predicción de hits"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Configuración de estrategias de cache
        self.cache_strategies = {
            'static_content': {
                'base_ttl': 86400 * 7,  # 1 semana
                'extension_factor': 1.5,
                'min_access_for_extension': 3
            },
            'dynamic_content': {
                'base_ttl': 3600,  # 1 hora
                'extension_factor': 2.0,
                'min_access_for_extension': 5
            },
            'personalized_content': {
                'base_ttl': 1800,  # 30 minutos
                'extension_factor': 1.2,
                'min_access_for_extension': 2
            },
            'analytical_content': {
                'base_ttl': 7200,  # 2 horas
                'extension_factor': 2.5,
                'min_access_for_extension': 3
            }
        }
        
        # ML features para predicción de cache hits
        self.ml_features_cache = {}
        
    async def get(self, 
                 key: str, 
                 update_stats: bool = True) -> Optional[Dict[str, Any]]:
        """Obtener entrada del cache con actualización de estadísticas"""
        
        try:
            cached_data = await self.redis.get(key)
            
            if cached_data:
                try:
                    cache_entry_data = json.loads(cached_data)
                    
                    if update_stats:
                        # Actualizar estadísticas de acceso
                        cache_entry_data['accessed_at'] = datetime.now().isoformat()
                        cache_entry_data['access_count'] += 1
                        
                        # Calcular nuevo hit prediction score
                        hit_score = self._calculate_hit_prediction_score(cache_entry_data)
                        cache_entry_data['hit_prediction_score'] = hit_score
                        
                        # Actualizar TTL dinámicamente si hay muchos accesos
                        await self._update_dynamic_ttl(key, cache_entry_data)
                        
                        # Guardar estadísticas actualizadas
                        await self.redis.set(key, json.dumps(cache_entry_data, default=str))
                    
                    return {
                        'cache_hit': True,
                        'data': cache_entry_data['value'],
                        'metadata': {
                            'cached_at': cache_entry_data['created_at'],
                            'access_count': cache_entry_data['access_count'],
                            'cost_saved_usd': cache_entry_data['cost_saved'],
                            'cache_strategy': cache_entry_data['cache_strategy'],
                            'hit_prediction_score': cache_entry_data.get('hit_prediction_score', 0.0)
                        }
                    }
                except json.JSONDecodeError:
                    # Cache corrupto, eliminar
                    await self.redis.delete(key)
                    return None
            
            return None
            
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, 
                 key: str, 
                 value: Any,
                 model_name: str,
                 prompt: str,
                 cost_saved: float,
                 custom_ttl: Optional[int] = None) -> bool:
        """Guardar entrada en cache con estrategia inteligente"""
        
        try:
            # Determinar estrategia de cache
            cache_strategy = self._determine_cache_strategy(prompt)
            
            # Calcular TTL
            ttl = custom_ttl or self._calculate_optimal_ttl(prompt, cache_strategy)
            
            # Calcular hit prediction score inicial
            hit_prediction_score = self._predict_cache_hit_probability(prompt, model_name)
            
            # Crear entrada de cache
            cache_entry = {
                'key': key,
                'value': value,
                'created_at': datetime.now().isoformat(),
                'accessed_at': datetime.now().isoformat(),
                'access_count': 0,
                'ttl_seconds': ttl,
                'model_name': model_name,
                'prompt_hash': hashlib.md5(prompt.encode()).hexdigest(),
                'cost_saved': cost_saved,
                'cache_strategy': cache_strategy,
                'hit_prediction_score': hit_prediction_score
            }
            
            # Guardar en Redis
            await self.redis.setex(key, ttl, json.dumps(cache_entry, default=str))
            
            # Mantener índices para analytics
            await self._update_cache_indexes(cache_entry)
            
            # Actualizar features de ML
            await self._update_ml_features(prompt, model_name, cache_strategy)
            
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def _determine_cache_strategy(self, prompt: str) -> str:
        """Determinar estrategia de cache basada en el contenido del prompt"""
        
        prompt_lower = prompt.lower()
        
        # Patrones para diferentes estrategias
        static_patterns = [
            'define', 'what is', 'explain', 'describe', 'list',
            'documentation', 'specification', 'reference'
        ]
        
        dynamic_patterns = [
            'analyze', 'current', 'recent', 'today', 'latest',
            'trend', 'update', 'news', 'real-time'
        ]
        
        personalized_patterns = [
            'my', 'i am', 'i have', 'i want', 'i need',
            'personal', 'customize', 'for me', 'based on my'
        ]
        
        analytical_patterns = [
            'calculate', 'compute', 'algorithm', 'formula',
            'statistics', 'data analysis', 'metrics'
        ]
        
        # Determinar estrategia basada en patrones
        if any(pattern in prompt_lower for pattern in personalized_patterns):
            return 'personalized_content'
        elif any(pattern in prompt_lower for pattern in dynamic_patterns):
            return 'dynamic_content'
        elif any(pattern in prompt_lower for pattern in analytical_patterns):
            return 'analytical_content'
        elif any(pattern in prompt_lower for pattern in static_patterns):
            return 'static_content'
        else:
            return 'dynamic_content'  # Default
    
    def _calculate_optimal_ttl(self, prompt: str, cache_strategy: str) -> int:
        """Calcular TTL óptimo basado en contenido y estrategia"""
        
        base_ttl = self.cache_strategies[cache_strategy]['base_ttl']
        
        # Factores de ajuste basados en contenido
        multiplier = 1.0
        
        # Prompts con fechas específicas -> TTL menor
        if any(pattern in prompt.lower() for pattern in ['today', 'this week', 'current']):
            multiplier *= 0.5
        
        # Prompts con información estática -> TTL mayor
        if any(pattern in prompt.lower() for pattern in ['definition', 'concept', 'theory']):
            multiplier *= 2.0
        
        # Prompts largos y complejos -> TTL mayor (costosos de regenerar)
        if len(prompt) > 1000:
            multiplier *= 1.5
        
        # Prompts con ejemplos -> TTL mayor (suelen reutilizarse)
        if 'example' in prompt.lower():
            multiplier *= 1.3
        
        return int(base_ttl * multiplier)
    
    def _predict_cache_hit_probability(self, prompt: str, model_name: str) -> float:
        """Predecir probabilidad de cache hit usando features simples"""
        
        features = self._extract_cache_features(prompt, model_name)
        
        # Modelo simple basado en reglas (en producción usar ML real)
        score = 0.5  # Base score
        
        # Factor por longitud del prompt
        if features['prompt_length'] > 500:
            score += 0.1  # Prompts largos se reutilizan más
        
        # Factor por complejidad
        if features['word_count'] > 100:
            score += 0.15
        
        # Factor por presencia de ejemplos
        if features['has_examples']:
            score += 0.2
        
        # Factor por tipo de contenido
        if features['content_type'] == 'static_content':
            score += 0.25
        elif features['content_type'] == 'analytical_content':
            score += 0.15
        
        # Factor por modelo (algunos modelos se cachean mejor)
        if 'gpt-4' in model_name:
            score += 0.1  # GPT-4 es costoso, mejor cachearlo
        
        return min(1.0, max(0.0, score))
    
    def _extract_cache_features(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Extraer features para ML de cache"""
        
        return {
            'prompt_length': len(prompt),
            'word_count': len(prompt.split()),
            'line_count': prompt.count('\n'),
            'has_examples': 'example' in prompt.lower(),
            'has_code': any(marker in prompt for marker in ['```', 'def ', 'class ', 'function']),
            'has_questions': '?' in prompt,
            'has_numbers': bool(re.search(r'\d+', prompt)),
            'content_type': self._determine_cache_strategy(prompt),
            'model_family': model_name.split('-')[0] if '-' in model_name else model_name,
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
    
    async def _update_dynamic_ttl(self, key: str, cache_entry_data: Dict[str, Any]):
        """Actualizar TTL dinámicamente basado en uso"""
        
        access_count = cache_entry_data['access_count']
        cache_strategy = cache_entry_data['cache_strategy']
        
        strategy_config = self.cache_strategies[cache_strategy]
        min_access = strategy_config['min_access_for_extension']
        extension_factor = strategy_config['extension_factor']
        
        # Si tiene suficientes accesos, extender TTL
        if access_count >= min_access:
            current_ttl = await self.redis.ttl(key)
            if current_ttl > 0:
                new_ttl = int(current_ttl * extension_factor)
                await self.redis.expire(key, new_ttl)
                cache_entry_data['ttl_seconds'] = new_ttl
    
    def _calculate_hit_prediction_score(self, cache_entry_data: Dict[str, Any]) -> float:
        """Calcular score de predicción de futuros hits"""
        
        access_count = cache_entry_data['access_count']
        created_at = datetime.fromisoformat(cache_entry_data['created_at'])
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        # Score basado en frecuencia de acceso
        if age_hours > 0:
            access_rate = access_count / age_hours
            base_score = min(1.0, access_rate / 2.0)  # Normalizar
        else:
            base_score = 0.5
        
        # Bonus por accesos recientes
        last_access = datetime.fromisoformat(cache_entry_data['accessed_at'])
        hours_since_access = (datetime.now() - last_access).total_seconds() / 3600
        
        if hours_since_access < 1:
            recency_bonus = 0.3
        elif hours_since_access < 6:
            recency_bonus = 0.1
        else:
            recency_bonus = 0.0
        
        return min(1.0, base_score + recency_bonus)
    
    async def _update_cache_indexes(self, cache_entry: Dict[str, Any]):
        """Actualizar índices para analytics"""
        
        # Índice por estrategia
        strategy = cache_entry['cache_strategy']
        await self.redis.sadd(f"cache_index:strategy:{strategy}", cache_entry['key'])
        
        # Índice por modelo
        model = cache_entry['model_name']
        await self.redis.sadd(f"cache_index:model:{model}", cache_entry['key'])
        
        # Índice por fecha (para limpieza)
        date_key = datetime.now().strftime('%Y%m%d')
        await self.redis.sadd(f"cache_index:date:{date_key}", cache_entry['key'])
    
    async def _update_ml_features(self, prompt: str, model_name: str, cache_strategy: str):
        """Actualizar features de ML para mejores predicciones"""
        
        features = self._extract_cache_features(prompt, model_name)
        
        # Guardar features para análisis posterior
        feature_key = f"ml_features:{datetime.now().strftime('%Y%m%d:%H')}"
        feature_data = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'cache_strategy': cache_strategy
        }
        
        await self.redis.lpush(feature_key, json.dumps(feature_data))
        await self.redis.expire(feature_key, 86400 * 7)  # 7 días
    
    async def get_cache_analytics(self) -> Dict[str, Any]:
        """Obtener analytics completos del cache"""
        
        try:
            # Estadísticas generales
            all_keys = await self.redis.keys("llm_cache:*")
            total_entries = len(all_keys)
            
            if total_entries == 0:
                return {'total_entries': 0, 'message': 'No cache entries found'}
            
            # Analizar entradas
            total_cost_saved = 0
            total_hits = 0
            strategy_distribution = {}
            model_distribution = {}
            hit_prediction_scores = []
            
            for key in all_keys[:1000]:  # Limitar para performance
                try:
                    data = await self.redis.get(key)
                    if data:
                        entry = json.loads(data)
                        total_cost_saved += entry.get('cost_saved', 0)
                        total_hits += entry.get('access_count', 0)
                        
                        strategy = entry.get('cache_strategy', 'unknown')
                        strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
                        
                        model = entry.get('model_name', 'unknown')
                        model_distribution[model] = model_distribution.get(model, 0) + 1
                        
                        hit_score = entry.get('hit_prediction_score', 0)
                        hit_prediction_scores.append(hit_score)
                        
                except:
                    continue
            
            # Calcular métricas
            avg_hit_prediction = np.mean(hit_prediction_scores) if hit_prediction_scores else 0
            hit_rate = (total_hits / max(total_entries, 1)) * 100
            
            return {
                'total_entries': total_entries,
                'total_cache_hits': total_hits,
                'estimated_hit_rate_percent': round(hit_rate, 2),
                'total_cost_saved_usd': round(total_cost_saved, 4),
                'monthly_projected_savings': round(total_cost_saved * 30, 2),
                'strategy_distribution': strategy_distribution,
                'model_distribution': model_distribution,
                'average_hit_prediction_score': round(avg_hit_prediction, 3),
                'cache_efficiency_metrics': {
                    'entries_with_multiple_hits': len([s for s in hit_prediction_scores if s > 0.5]),
                    'high_value_entries': len([s for s in hit_prediction_scores if s > 0.8]),
                    'cache_utilization_score': round(avg_hit_prediction * hit_rate / 100, 3)
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def optimize_cache_storage(self) -> Dict[str, Any]:
        """Optimizar almacenamiento de cache eliminando entradas de bajo valor"""
        
        try:
            all_keys = await self.redis.keys("llm_cache:*")
            
            candidates_for_removal = []
            total_examined = 0
            
            for key in all_keys:
                try:
                    data = await self.redis.get(key)
                    if data:
                        entry = json.loads(data)
                        total_examined += 1
                        
                        # Criterios para eliminación
                        hit_score = entry.get('hit_prediction_score', 0)
                        access_count = entry.get('access_count', 0)
                        cost_saved = entry.get('cost_saved', 0)
                        
                        # Eliminar si:
                        # - Muy baja probabilidad de hit futuro
                        # - No ha sido accedido nunca
                        # - Costo ahorrado muy bajo
                        if (hit_score < 0.2 and access_count == 0) or cost_saved < 0.0001:
                            candidates_for_removal.append(key)
                            
                except:
                    # Entrada corrupta, marcar para eliminación
                    candidates_for_removal.append(key)
            
            # Eliminar entradas candidatas
            removed_count = 0
            if candidates_for_removal:
                removed_count = await self.redis.delete(*candidates_for_removal)
            
            return {
                'optimization_completed': True,
                'total_entries_examined': total_examined,
                'entries_removed': removed_count,
                'storage_freed_percent': round((removed_count / max(total_examined, 1)) * 100, 2),
                'remaining_entries': total_examined - removed_count
            }
            
        except Exception as e:
            return {'optimization_completed': False, 'error': str(e)}
    
    async def preload_high_value_cache(self, popular_prompts: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Pre-cargar cache con prompts de alto valor"""
        
        preloaded = 0
        errors = 0
        
        for prompt, model_name in popular_prompts:
            try:
                # Generar clave de cache
                cache_key = self._generate_cache_key(prompt, model_name)
                
                # Verificar si ya existe
                existing = await self.redis.get(cache_key)
                if not existing:
                    # Simular carga (en producción, hacer request real)
                    # Por ahora, crear entrada placeholder
                    await self.set(
                        key=cache_key,
                        value="[PRELOADED_PLACEHOLDER]",
                        model_name=model_name,
                        prompt=prompt,
                        cost_saved=0.01  # Estimado
                    )
                    preloaded += 1
                    
            except Exception as e:
                errors += 1
                print(f"Error preloading cache for prompt: {e}")
        
        return {
            'preload_completed': True,
            'prompts_processed': len(popular_prompts),
            'entries_preloaded': preloaded,
            'errors': errors
        }
    
    def _generate_cache_key(self, prompt: str, model_name: str, **kwargs) -> str:
        """Generar clave de cache consistente"""
        
        cache_data = {
            'model': model_name,
            'prompt': prompt.strip().lower(),
            **kwargs
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"llm_cache:{hashlib.sha256(cache_string.encode()).hexdigest()}"
```

### 3. API Principal del Sistema

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import asyncio
import redis.asyncio as redis

from ..optimization.prompt_optimizer import AdvancedPromptOptimizer
from ..optimization.cache_manager import SmartCacheManager
from ..optimization.cost_monitor import CostMonitor
from ..optimization.model_selector import IntelligentModelSelector, TaskRequirements

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Cost Optimization API",
    description="Advanced cost optimization and caching system for LLMs",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class OptimizePromptRequest(BaseModel):
    prompt: str
    optimization_level: str = "standard"  # conservative, standard, aggressive
    preserve_structure: bool = True

class CacheRequest(BaseModel):
    key: str
    model_name: str
    prompt: str
    response: Any
    input_tokens: int = 0
    output_tokens: int = 0

class ModelSelectionRequest(BaseModel):
    task_type: str
    complexity_level: str = "medium"
    required_capabilities: List[str] = []
    max_acceptable_cost: float = 1.0
    quality_threshold: float = 0.8
    speed_requirement: str = "medium"
    context_length_needed: int = 4000
    estimated_input_tokens: int = 1000
    estimated_output_tokens: int = 500

# Global components
redis_client = None
prompt_optimizer = None
cache_manager = None
cost_monitor = None
model_selector = None

@app.on_event("startup")
async def startup_event():
    """Inicializar servicios al arrancar"""
    global redis_client, prompt_optimizer, cache_manager, cost_monitor, model_selector
    
    logger.info("Initializing Cost Optimization API...")
    
    # Inicializar Redis
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    
    # Inicializar componentes
    prompt_optimizer = AdvancedPromptOptimizer()
    cache_manager = SmartCacheManager(redis_client)
    cost_monitor = CostMonitor(redis_client)
    model_selector = IntelligentModelSelector()
    
    logger.info("API ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar"""
    if redis_client:
        await redis_client.close()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "LLM Cost Optimization API",
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "features": [
            "prompt_optimization",
            "intelligent_caching",
            "cost_monitoring", 
            "model_selection"
        ]
    }

@app.post("/optimize/prompt")
async def optimize_prompt(request: OptimizePromptRequest):
    """Optimizar prompt para reducir tokens y costos"""
    
    try:
        result = await prompt_optimizer.optimize_prompt(
            prompt=request.prompt,
            optimization_level=request.optimization_level,
            preserve_structure=request.preserve_structure
        )
        
        return {
            "success": True,
            "optimization_result": {
                "original_prompt": result.original_prompt,
                "optimized_prompt": result.optimized_prompt,
                "metrics": {
                    "original_tokens": result.original_tokens,
                    "optimized_tokens": result.optimized_tokens,
                    "tokens_saved": result.original_tokens - result.optimized_tokens,
                    "reduction_percentage": result.reduction_percentage,
                    "estimated_savings_usd": result.estimated_savings_usd
                },
                "optimizations_applied": result.optimizations_applied
            }
        }
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/batch-prompts")
async def batch_optimize_prompts(prompts: List[str], optimization_level: str = "standard"):
    """Optimizar múltiples prompts en lote"""
    
    try:
        result = await prompt_optimizer.batch_optimize(prompts, optimization_level)
        
        return {
            "success": True,
            "batch_result": result
        }
        
    except Exception as e:
        logger.error(f"Error batch optimizing prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/prompt-efficiency")
async def analyze_prompt_efficiency(prompt: str):
    """Analizar eficiencia de un prompt sin optimizar"""
    
    try:
        analysis = prompt_optimizer.analyze_prompt_efficiency(prompt)
        
        return {
            "success": True,
            "efficiency_analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing prompt efficiency: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/set")
async def set_cache(request: CacheRequest):
    """Guardar respuesta en cache inteligente"""
    
    try:
        # Calcular costo ahorrado estimado
        cost_saved = 0.01  # Placeholder - calcular basado en tokens reales
        
        cache_key = cache_manager._generate_cache_key(
            request.prompt, 
            request.model_name
        )
        
        success = await cache_manager.set(
            key=cache_key,
            value=request.response,
            model_name=request.model_name,
            prompt=request.prompt,
            cost_saved=cost_saved
        )
        
        return {
            "success": success,
            "cache_key": cache_key,
            "cost_saved_usd": cost_saved
        }
        
    except Exception as e:
        logger.error(f"Error setting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/get/{cache_key}")
async def get_cache(cache_key: str):
    """Obtener respuesta del cache"""
    
    try:
        result = await cache_manager.get(cache_key)
        
        if result:
            return {
                "success": True,
                "cache_hit": True,
                "data": result['data'],
                "metadata": result['metadata']
            }
        else:
            return {
                "success": True,
                "cache_hit": False,
                "message": "Cache miss"
            }
            
    except Exception as e:
        logger.error(f"Error getting cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/analytics")
async def get_cache_analytics():
    """Obtener analytics del cache"""
    
    try:
        analytics = await cache_manager.get_cache_analytics()
        
        return {
            "success": True,
            "cache_analytics": analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting cache analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/optimize")
async def optimize_cache_storage():
    """Optimizar almacenamiento del cache"""
    
    try:
        result = await cache_manager.optimize_cache_storage()
        
        return {
            "success": True,
            "optimization_result": result
        }
        
    except Exception as e:
        logger.error(f"Error optimizing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/select")
async def select_optimal_model(request: ModelSelectionRequest):
    """Seleccionar modelo óptimo basado en requerimientos"""
    
    try:
        requirements = TaskRequirements(
            task_type=request.task_type,
            complexity_level=request.complexity_level,
            required_capabilities=request.required_capabilities,
            max_acceptable_cost=request.max_acceptable_cost,
            quality_threshold=request.quality_threshold,
            speed_requirement=request.speed_requirement,
            context_length_needed=request.context_length_needed
        )
        
        result = model_selector.select_optimal_model(
            requirements,
            request.estimated_input_tokens,
            request.estimated_output_tokens
        )
        
        return {
            "success": result['success'],
            "selection_result": result
        }
        
    except Exception as e:
        logger.error(f"Error selecting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cost/analytics")
async def get_cost_analytics(period: str = "24h", granularity: str = "hour"):
    """Obtener analytics de costos"""
    
    try:
        analytics = await cost_monitor.get_cost_analytics(period, granularity)
        
        return {
            "success": True,
            "cost_analytics": analytics
        }
        
    except Exception as e:
        logger.error(f"Error getting cost analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cost/track")
async def track_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    user_id: Optional[str] = None,
    request_type: str = "api"
):
    """Trackear costo de una request"""
    
    try:
        result = await cost_monitor.track_request(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_id=user_id,
            request_type=request_type
        )
        
        return {
            "success": True,
            "tracking_result": result
        }
        
    except Exception as e:
        logger.error(f"Error tracking cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """Obtener métricas para dashboard"""
    
    try:
        # Obtener métricas de todos los componentes
        cache_analytics = await cache_manager.get_cache_analytics()
        cost_analytics = await cost_monitor.get_cost_analytics("24h")
        
        # Métricas consolidadas
        dashboard_metrics = {
            "timestamp": datetime.now().isoformat(),
            "cache_metrics": {
                "total_entries": cache_analytics.get('total_entries', 0),
                "hit_rate_percent": cache_analytics.get('estimated_hit_rate_percent', 0),
                "total_cost_saved": cache_analytics.get('total_cost_saved_usd', 0),
                "monthly_projected_savings": cache_analytics.get('monthly_projected_savings', 0)
            },
            "cost_metrics": {
                "total_requests_24h": cost_analytics.get('summary', {}).get('total_requests', 0),
                "total_cost_24h": cost_analytics.get('summary', {}).get('total_cost_usd', 0),
                "avg_cost_per_request": cost_analytics.get('summary', {}).get('avg_cost_per_request', 0),
                "projected_monthly_cost": cost_analytics.get('projected_monthly_cost', 0)
            },
            "optimization_metrics": {
                "prompts_optimized_today": 0,  # TODO: implementar tracking
                "avg_token_reduction": 0,      # TODO: implementar tracking
                "total_savings_from_optimization": 0  # TODO: implementar tracking
            }
        }
        
        return {
            "success": True,
            "dashboard_metrics": dashboard_metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

## 🧪 Ejercicios Prácticos

### Ejercicio 1: Optimización Básica
1. Configurar el entorno Docker
2. Probar optimización de prompts con diferentes niveles
3. Medir reducción de tokens y costos

### Ejercicio 2: Cache Inteligente
1. Implementar estrategias de caching
2. Probar hit rates con diferentes tipos de prompts
3. Optimizar configuración de TTL

### Ejercicio 3: Selección de Modelos
1. Configurar múltiples modelos con precios
2. Probar selección automática por tareas
3. Analizar ahorros de costo

### Ejercicio 4: Monitoreo Completo
1. Implementar tracking de costos en tiempo real
2. Configurar alertas por presupuesto
3. Crear dashboard de analytics

## 📊 Métricas de Éxito

- **Reducción de tokens**: >15% promedio
- **Hit rate de cache**: >60%
- **Reducción de costos**: >25%
- **Tiempo de respuesta API**: <100ms
- **Precisión de selección de modelos**: >90%

## 🔄 Próximos Pasos

1. **Lección 2**: Performance y Escalabilidad
2. **Laboratorio 6**: Load Balancing y Rate Limiting
3. **Integración**: Con sistemas de producción
4. **ML avanzado**: Para predicción de cache y optimización

## 📚 Recursos Adicionales

- [OpenAI Token Optimization](https://help.openai.com/en/articles/4936856)
- [Redis Caching Strategies](https://redis.io/docs/manual/patterns/)
- [FastAPI Performance Guide](https://fastapi.tiangolo.com/advanced/performance/)
- [LLM Cost Management Best Practices](https://www.anthropic.com/news/cost-effective-llm-deployment)
