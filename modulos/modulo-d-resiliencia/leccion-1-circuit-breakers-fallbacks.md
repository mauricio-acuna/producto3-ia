# 🛡️ Lección 1: Circuit Breakers y Fallback Strategies

## 🎯 Objetivos de la Lección

Al finalizar esta lección, serás capaz de:
- Implementar circuit breakers robustos para LLMs
- Diseñar estrategias de fallback inteligentes
- Configurar timeout management avanzado
- Crear sistemas de retry con backoff exponencial
- Implementar bulkhead patterns para aislamiento
- Gestionar degradación gradual de servicios

## 🔧 Fundamentos de Resilience

### 1. Circuit Breaker Avanzado para LLMs

```python
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from abc import ABC, abstractmethod

class CircuitState(Enum):
    CLOSED = "closed"      # Funcionamiento normal
    OPEN = "open"          # Circuit abierto - no permite requests
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Fallos para abrir circuit
    success_threshold: int = 3          # Éxitos para cerrar circuit
    timeout_seconds: int = 60           # Tiempo antes de pasar a half-open
    slow_call_threshold: float = 5.0    # Tiempo considerado "lento" (segundos)
    slow_call_rate_threshold: float = 0.5  # % de calls lentas para abrir
    minimum_throughput: int = 10        # Mínimo de calls para evaluar
    sliding_window_size: int = 100      # Tamaño de ventana deslizante
    max_wait_duration: float = 60.0     # Máximo tiempo de espera

@dataclass
class CallResult:
    success: bool
    duration_seconds: float
    timestamp: datetime
    error: Optional[str] = None
    response_size: Optional[int] = None

class AdvancedCircuitBreaker:
    """Circuit Breaker avanzado con múltiples criterios de fallo"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.last_success_time = None
        
        # Ventana deslizante de resultados
        self.call_results: List[CallResult] = []
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        # Métricas
        self.total_calls = 0
        self.total_failures = 0
        self.total_slow_calls = 0
        
        # Logging
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        
        # Estado de half-open
        self.half_open_calls = 0
        
    async def call(self, func: Callable, *args, fallback: Optional[Callable] = None, **kwargs) -> Any:
        """Ejecutar función con circuit breaker"""
        
        self.total_calls += 1
        
        # Verificar estado del circuit
        if not self._should_allow_call():
            self.logger.warning(f"Circuit breaker {self.name} is OPEN - rejecting call")
            
            if fallback:
                self.logger.info(f"Executing fallback for {self.name}")
                return await self._execute_fallback(fallback, *args, **kwargs)
            else:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker {self.name} is open. Last failure: {self.last_failure_time}"
                )
        
        # Ejecutar la función
        start_time = time.time()
        
        try:
            # Aplicar timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.max_wait_duration
            )
            
            duration = time.time() - start_time
            
            # Registrar resultado exitoso
            await self._record_success(duration)
            
            return result
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            await self._record_failure(duration, "Timeout")
            
            if fallback:
                self.logger.info(f"Timeout occurred, executing fallback for {self.name}")
                return await self._execute_fallback(fallback, *args, **kwargs)
            else:
                raise CircuitBreakerTimeoutException(
                    f"Circuit breaker {self.name} timeout after {duration:.2f}s"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            await self._record_failure(duration, str(e))
            
            if fallback:
                self.logger.info(f"Error occurred, executing fallback for {self.name}: {e}")
                return await self._execute_fallback(fallback, *args, **kwargs)
            else:
                raise
    
    def _should_allow_call(self) -> bool:
        """Determinar si permitir la llamada"""
        
        current_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Verificar si es tiempo de pasar a half-open
            if (self.last_failure_time and 
                (current_time - self.last_failure_time).total_seconds() >= self.config.timeout_seconds):
                self._transition_to_half_open()
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            # En half-open permitimos un número limitado de calls
            return self.half_open_calls < self.config.success_threshold
        
        return False
    
    async def _execute_fallback(self, fallback: Callable, *args, **kwargs) -> Any:
        """Ejecutar función de fallback"""
        
        try:
            if asyncio.iscoroutinefunction(fallback):
                return await fallback(*args, **kwargs)
            else:
                return fallback(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Fallback failed for {self.name}: {e}")
            raise FallbackFailedException(f"Both primary and fallback failed: {e}")
    
    async def _record_success(self, duration: float):
        """Registrar llamada exitosa"""
        
        now = datetime.now()
        
        # Determinar si es una llamada lenta
        is_slow = duration > self.config.slow_call_threshold
        
        call_result = CallResult(
            success=True,
            duration_seconds=duration,
            timestamp=now,
            error=None
        )
        
        self._add_call_result(call_result)
        
        if is_slow:
            self.total_slow_calls += 1
        
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.last_success_time = now
        
        # Evaluar transición de estado
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            
            if self.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()
        
        # Verificar si deberíamos abrir por slow calls
        elif self.state == CircuitState.CLOSED:
            self._evaluate_slow_call_rate()
        
        self.logger.debug(f"Success recorded for {self.name}: {duration:.3f}s")
    
    async def _record_failure(self, duration: float, error: str):
        """Registrar llamada fallida"""
        
        now = datetime.now()
        
        call_result = CallResult(
            success=False,
            duration_seconds=duration,
            timestamp=now,
            error=error
        )
        
        self._add_call_result(call_result)
        
        self.total_failures += 1
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_failure_time = now
        
        # Evaluar transición de estado
        if self.state == CircuitState.CLOSED:
            if self.consecutive_failures >= self.config.failure_threshold:
                self._transition_to_open()
                
        elif self.state == CircuitState.HALF_OPEN:
            # Cualquier fallo en half-open vuelve a open
            self._transition_to_open()
        
        self.logger.warning(f"Failure recorded for {self.name}: {error}")
    
    def _add_call_result(self, call_result: CallResult):
        """Añadir resultado a la ventana deslizante"""
        
        self.call_results.append(call_result)
        
        # Mantener tamaño de ventana
        if len(self.call_results) > self.config.sliding_window_size:
            self.call_results.pop(0)
    
    def _evaluate_slow_call_rate(self):
        """Evaluar si abrir circuit por slow calls"""
        
        if len(self.call_results) < self.config.minimum_throughput:
            return
        
        # Calcular rate de slow calls en ventana reciente
        recent_calls = self.call_results[-self.config.minimum_throughput:]
        slow_calls = [c for c in recent_calls if c.duration_seconds > self.config.slow_call_threshold]
        
        slow_call_rate = len(slow_calls) / len(recent_calls)
        
        if slow_call_rate >= self.config.slow_call_rate_threshold:
            self.logger.warning(
                f"Opening circuit {self.name} due to slow call rate: {slow_call_rate:.2%}"
            )
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transición a estado OPEN"""
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        self.logger.error(f"Circuit breaker {self.name} opened")
    
    def _transition_to_half_open(self):
        """Transición a estado HALF_OPEN"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned to half-open")
    
    def _transition_to_closed(self):
        """Transición a estado CLOSED"""
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
        self.consecutive_failures = 0
        self.logger.info(f"Circuit breaker {self.name} closed")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del circuit breaker"""
        
        if not self.call_results:
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.total_calls,
                'failure_rate': 0.0,
                'slow_call_rate': 0.0,
                'avg_response_time': 0.0,
                'consecutive_failures': self.consecutive_failures,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None
            }
        
        # Calcular métricas de la ventana reciente
        recent_window = self.call_results[-50:] if len(self.call_results) >= 50 else self.call_results
        
        failures = [c for c in recent_window if not c.success]
        slow_calls = [c for c in recent_window if c.duration_seconds > self.config.slow_call_threshold]
        
        failure_rate = len(failures) / len(recent_window) if recent_window else 0.0
        slow_call_rate = len(slow_calls) / len(recent_window) if recent_window else 0.0
        avg_response_time = statistics.mean([c.duration_seconds for c in recent_window])
        
        return {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'failure_rate': round(failure_rate, 3),
            'slow_call_rate': round(slow_call_rate, 3),
            'avg_response_time': round(avg_response_time, 3),
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
            'window_size': len(self.call_results),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'slow_call_threshold': self.config.slow_call_threshold
            }
        }
    
    def reset(self):
        """Reset del circuit breaker"""
        self.state = CircuitState.CLOSED
        self.call_results.clear()
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.logger.info(f"Circuit breaker {self.name} reset")

# Excepciones personalizadas
class CircuitBreakerException(Exception):
    """Excepción base para circuit breaker"""
    pass

class CircuitBreakerOpenException(CircuitBreakerException):
    """Circuit breaker está abierto"""
    pass

class CircuitBreakerTimeoutException(CircuitBreakerException):
    """Timeout en circuit breaker"""
    pass

class FallbackFailedException(CircuitBreakerException):
    """Falló tanto la función principal como el fallback"""
    pass
```

### 2. Fallback Strategies Inteligentes

```python
from typing import Any, Dict, List, Optional, Callable, Union
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import hashlib

class FallbackStrategy(ABC):
    """Estrategia base para fallbacks"""
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Ejecutar estrategia de fallback"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Obtener prioridad (menor número = mayor prioridad)"""
        pass

@dataclass
class LLMFallbackContext:
    original_prompt: str
    model_requested: str
    max_tokens: int
    temperature: float
    user_id: Optional[str] = None
    attempt_number: int = 1
    original_error: Optional[str] = None

class CachedResponseFallback(FallbackStrategy):
    """Fallback usando respuestas cacheadas similares"""
    
    def __init__(self, cache_client, similarity_threshold: float = 0.8):
        self.cache_client = cache_client
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger("fallback.cached")
    
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Buscar respuesta similar en cache"""
        
        llm_context = LLMFallbackContext(**context)
        
        # Buscar respuestas similares en cache
        similar_responses = await self._find_similar_cached_responses(
            llm_context.original_prompt
        )
        
        if similar_responses:
            best_match = similar_responses[0]
            
            self.logger.info(
                f"Using cached fallback for prompt similarity: {best_match['similarity']:.2f}"
            )
            
            return {
                'text': best_match['response'],
                'fallback_type': 'cached_response',
                'similarity_score': best_match['similarity'],
                'cached_prompt': best_match['prompt'],
                'tokens_used': len(best_match['response'].split())
            }
        
        return None
    
    async def _find_similar_cached_responses(self, prompt: str) -> List[Dict[str, Any]]:
        """Encontrar respuestas similares en cache"""
        
        # Implementación simplificada - en producción usar embeddings
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        try:
            # Buscar por hash parcial (simulación de similaridad)
            cached_keys = await self.cache_client.keys(f"llm_response:*{prompt_hash[:4]}*")
            
            similar_responses = []
            
            for key in cached_keys[:5]:  # Limitar búsqueda
                cached_data = await self.cache_client.get(key)
                if cached_data:
                    data = json.loads(cached_data)
                    similarity = self._calculate_similarity(prompt, data.get('prompt', ''))
                    
                    if similarity >= self.similarity_threshold:
                        similar_responses.append({
                            'prompt': data.get('prompt'),
                            'response': data.get('response'),
                            'similarity': similarity
                        })
            
            # Ordenar por similaridad
            return sorted(similar_responses, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error finding similar cached responses: {e}")
            return []
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcular similaridad simple entre textos"""
        
        # Implementación básica - en producción usar embeddings semánticos
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_priority(self) -> int:
        return 1  # Alta prioridad

class AlternativeModelFallback(FallbackStrategy):
    """Fallback usando modelo alternativo más simple"""
    
    def __init__(self, alternative_models: List[str], llm_client):
        self.alternative_models = alternative_models
        self.llm_client = llm_client
        self.logger = logging.getLogger("fallback.alternative_model")
    
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Usar modelo alternativo"""
        
        llm_context = LLMFallbackContext(**context)
        
        # Intentar con modelos alternativos en orden
        for alt_model in self.alternative_models:
            if alt_model == llm_context.model_requested:
                continue  # Skip el modelo que falló
            
            try:
                self.logger.info(f"Trying alternative model: {alt_model}")
                
                # Ajustar parámetros para modelo más simple
                adjusted_max_tokens = min(llm_context.max_tokens, 100)
                adjusted_temperature = min(llm_context.temperature, 0.7)
                
                response = await self.llm_client.generate(
                    prompt=llm_context.original_prompt,
                    model=alt_model,
                    max_tokens=adjusted_max_tokens,
                    temperature=adjusted_temperature
                )
                
                return {
                    'text': response['text'],
                    'fallback_type': 'alternative_model',
                    'model_used': alt_model,
                    'original_model': llm_context.model_requested,
                    'tokens_used': response.get('tokens_used', 0)
                }
                
            except Exception as e:
                self.logger.warning(f"Alternative model {alt_model} also failed: {e}")
                continue
        
        return None
    
    def get_priority(self) -> int:
        return 2  # Prioridad media

class SimplifiedPromptFallback(FallbackStrategy):
    """Fallback simplificando el prompt original"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.logger = logging.getLogger("fallback.simplified_prompt")
    
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Simplificar prompt y reintentar"""
        
        llm_context = LLMFallbackContext(**context)
        
        # Simplificar el prompt
        simplified_prompt = self._simplify_prompt(llm_context.original_prompt)
        
        if simplified_prompt == llm_context.original_prompt:
            return None  # No se pudo simplificar
        
        try:
            self.logger.info("Trying with simplified prompt")
            
            response = await self.llm_client.generate(
                prompt=simplified_prompt,
                model=llm_context.model_requested,
                max_tokens=min(llm_context.max_tokens, 50),  # Reducir tokens
                temperature=0.3  # Reducir creatividad
            )
            
            return {
                'text': response['text'],
                'fallback_type': 'simplified_prompt',
                'original_prompt': llm_context.original_prompt,
                'simplified_prompt': simplified_prompt,
                'tokens_used': response.get('tokens_used', 0)
            }
            
        except Exception as e:
            self.logger.warning(f"Simplified prompt also failed: {e}")
            return None
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Simplificar prompt quitando detalles"""
        
        # Reglas de simplificación
        lines = prompt.split('\n')
        
        # Quitar líneas muy largas
        simplified_lines = [line for line in lines if len(line) <= 100]
        
        # Tomar solo las primeras oraciones importantes
        if len(simplified_lines) > 3:
            simplified_lines = simplified_lines[:3]
        
        simplified = '\n'.join(simplified_lines).strip()
        
        # Limitar longitud total
        if len(simplified) > 200:
            simplified = simplified[:200] + "..."
        
        return simplified if simplified != prompt else prompt
    
    def get_priority(self) -> int:
        return 3  # Prioridad baja

class StaticResponseFallback(FallbackStrategy):
    """Fallback con respuesta estática apropiada"""
    
    def __init__(self):
        self.logger = logging.getLogger("fallback.static")
        
        # Respuestas por tipo de prompt
        self.static_responses = {
            'question': "I apologize, but I'm currently unable to process your question. Please try again later.",
            'creative': "I'm sorry, I'm experiencing technical difficulties with creative tasks. Please try again later.",
            'analysis': "I'm currently unable to perform detailed analysis. Please try again later.",
            'code': "I'm unable to generate code at the moment. Please try again later.",
            'default': "I apologize for the inconvenience. The service is temporarily unavailable. Please try again later."
        }
    
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Devolver respuesta estática apropiada"""
        
        llm_context = LLMFallbackContext(**context)
        
        # Detectar tipo de prompt
        prompt_type = self._detect_prompt_type(llm_context.original_prompt)
        
        response_text = self.static_responses.get(prompt_type, self.static_responses['default'])
        
        self.logger.info(f"Using static fallback response for prompt type: {prompt_type}")
        
        return {
            'text': response_text,
            'fallback_type': 'static_response',
            'prompt_type': prompt_type,
            'tokens_used': len(response_text.split())
        }
    
    def _detect_prompt_type(self, prompt: str) -> str:
        """Detectar tipo de prompt"""
        
        prompt_lower = prompt.lower()
        
        # Palabras clave para diferentes tipos
        if any(word in prompt_lower for word in ['?', 'what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        elif any(word in prompt_lower for word in ['write', 'create', 'story', 'poem', 'creative']):
            return 'creative'
        elif any(word in prompt_lower for word in ['analyze', 'compare', 'evaluate', 'assess']):
            return 'analysis'
        elif any(word in prompt_lower for word in ['code', 'function', 'class', 'script', 'program']):
            return 'code'
        else:
            return 'default'
    
    def get_priority(self) -> int:
        return 99  # Última opción

class IntelligentFallbackManager:
    """Gestor inteligente de estrategias de fallback"""
    
    def __init__(self):
        self.strategies: List[FallbackStrategy] = []
        self.logger = logging.getLogger("fallback.manager")
        
        # Métricas de uso
        self.strategy_usage = {}
        self.strategy_success_rate = {}
    
    def add_strategy(self, strategy: FallbackStrategy):
        """Añadir estrategia de fallback"""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.get_priority())
        
        strategy_name = strategy.__class__.__name__
        self.strategy_usage[strategy_name] = 0
        self.strategy_success_rate[strategy_name] = []
    
    async def execute_fallback(self, context: Dict[str, Any]) -> Any:
        """Ejecutar fallback usando la mejor estrategia disponible"""
        
        self.logger.info("Executing fallback strategies")
        
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            
            try:
                self.logger.debug(f"Trying fallback strategy: {strategy_name}")
                
                result = await strategy.execute(context)
                
                if result is not None:
                    # Registrar éxito
                    self.strategy_usage[strategy_name] += 1
                    self.strategy_success_rate[strategy_name].append(True)
                    
                    # Mantener solo últimos 100 resultados
                    if len(self.strategy_success_rate[strategy_name]) > 100:
                        self.strategy_success_rate[strategy_name] = self.strategy_success_rate[strategy_name][-100:]
                    
                    self.logger.info(f"Fallback successful using {strategy_name}")
                    
                    result['fallback_strategy'] = strategy_name
                    result['fallback_priority'] = strategy.get_priority()
                    
                    return result
                    
            except Exception as e:
                # Registrar fallo
                self.strategy_success_rate[strategy_name].append(False)
                self.logger.warning(f"Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # Si todas las estrategias fallaron
        self.logger.error("All fallback strategies failed")
        
        return {
            'text': "I apologize, but I'm currently experiencing technical difficulties. Please try again later.",
            'fallback_type': 'emergency_response',
            'fallback_strategy': 'emergency',
            'tokens_used': 15
        }
    
    def get_fallback_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de fallback"""
        
        metrics = {
            'total_strategies': len(self.strategies),
            'strategy_stats': {}
        }
        
        for strategy_name, usage_count in self.strategy_usage.items():
            success_rates = self.strategy_success_rate.get(strategy_name, [])
            success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
            
            metrics['strategy_stats'][strategy_name] = {
                'usage_count': usage_count,
                'success_rate': round(success_rate, 3),
                'priority': next(s.get_priority() for s in self.strategies if s.__class__.__name__ == strategy_name)
            }
        
        return metrics
```

### 3. Retry Logic con Backoff Exponencial

```python
import asyncio
import random
import time
import logging
from typing import Callable, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class RetryStrategy(Enum):
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    ADAPTIVE = "adaptive"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    jitter_range: float = 0.1
    backoff_multiplier: float = 2.0
    retryable_exceptions: List[type] = None

class RetryableError(Exception):
    """Error que puede ser reintentado"""
    pass

class NonRetryableError(Exception):
    """Error que NO debe ser reintentado"""
    pass

class IntelligentRetryManager:
    """Gestor inteligente de reintentos con múltiples estrategias"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("retry.manager")
        
        # Métricas
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_after_retries = 0
        
        # Configurar excepciones reintentables por defecto
        if self.config.retryable_exceptions is None:
            self.config.retryable_exceptions = [
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                RetryableError
            ]
    
    async def execute_with_retry(self, 
                               func: Callable,
                               *args,
                               custom_config: Optional[RetryConfig] = None,
                               context: Optional[Dict[str, Any]] = None,
                               **kwargs) -> Any:
        """Ejecutar función con retry logic"""
        
        config = custom_config or self.config
        context = context or {}
        
        last_exception = None
        
        for attempt in range(1, config.max_attempts + 1):
            self.total_attempts += 1
            
            try:
                self.logger.debug(f"Attempt {attempt}/{config.max_attempts}")
                
                # Ejecutar función
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Éxito
                if attempt > 1:
                    self.successful_retries += 1
                    self.logger.info(f"Success after {attempt} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Verificar si es reintentable
                if not self._is_retryable_exception(e, config):
                    self.logger.error(f"Non-retryable exception: {e}")
                    raise e
                
                # Si es el último intento, lanzar excepción
                if attempt >= config.max_attempts:
                    self.failed_after_retries += 1
                    self.logger.error(f"Failed after {config.max_attempts} attempts: {e}")
                    raise e
                
                # Calcular delay
                delay = self._calculate_delay(attempt, config, context)
                
                self.logger.warning(
                    f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                )
                
                # Esperar antes del siguiente intento
                await asyncio.sleep(delay)
        
        # No debería llegar aquí, pero por seguridad
        raise last_exception
    
    def _is_retryable_exception(self, exception: Exception, config: RetryConfig) -> bool:
        """Verificar si una excepción es reintentable"""
        
        # Excepciones explícitamente no reintentables
        if isinstance(exception, NonRetryableError):
            return False
        
        # Verificar lista de excepciones reintentables
        return any(isinstance(exception, exc_type) for exc_type in config.retryable_exceptions)
    
    def _calculate_delay(self, 
                        attempt: int, 
                        config: RetryConfig,
                        context: Dict[str, Any]) -> float:
        """Calcular delay para el próximo intento"""
        
        if config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
            
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** (attempt - 1))
            
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt
            
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = config.base_delay * self._fibonacci(attempt)
            
        elif config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(attempt, context, config)
            
        else:
            delay = config.base_delay
        
        # Aplicar límite máximo
        delay = min(delay, config.max_delay)
        
        # Aplicar jitter si está habilitado
        if config.jitter:
            jitter_amount = delay * config.jitter_range
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calcular número de Fibonacci"""
        if n <= 1:
            return 1
        elif n == 2:
            return 1
        else:
            a, b = 1, 1
            for _ in range(3, n + 1):
                a, b = b, a + b
            return b
    
    def _calculate_adaptive_delay(self, 
                                 attempt: int,
                                 context: Dict[str, Any],
                                 config: RetryConfig) -> float:
        """Calcular delay adaptivo basado en contexto"""
        
        base_delay = config.base_delay
        
        # Factores adaptativos
        load_factor = context.get('system_load', 1.0)
        error_rate = context.get('recent_error_rate', 0.1)
        response_time = context.get('avg_response_time', 1.0)
        
        # Ajustar delay basado en condiciones del sistema
        if load_factor > 0.8:  # Sistema con alta carga
            base_delay *= 2.0
        
        if error_rate > 0.3:  # Alta tasa de errores
            base_delay *= 1.5
        
        if response_time > 5.0:  # Respuestas lentas
            base_delay *= 1.3
        
        # Aplicar backoff exponencial sobre el delay adaptado
        delay = base_delay * (config.backoff_multiplier ** (attempt - 1))
        
        return delay
    
    def get_retry_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de retry"""
        
        total_operations = self.successful_retries + self.failed_after_retries
        
        if total_operations == 0:
            success_rate = 1.0
        else:
            success_rate = self.successful_retries / total_operations
        
        return {
            'total_attempts': self.total_attempts,
            'successful_retries': self.successful_retries,
            'failed_after_retries': self.failed_after_retries,
            'retry_success_rate': round(success_rate, 3),
            'avg_attempts_per_operation': round(
                self.total_attempts / max(total_operations, 1), 2
            ),
            'config': {
                'max_attempts': self.config.max_attempts,
                'strategy': self.config.strategy.value,
                'base_delay': self.config.base_delay,
                'max_delay': self.config.max_delay
            }
        }

# Decorador para retry automático
def retry_on_failure(config: RetryConfig = None):
    """Decorador para añadir retry automático a funciones"""
    
    def decorator(func: Callable) -> Callable:
        retry_manager = IntelligentRetryManager(config)
        
        async def async_wrapper(*args, **kwargs):
            return await retry_manager.execute_with_retry(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(retry_manager.execute_with_retry(func, *args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
```

### 4. Sistema Integrado de Resilience

```python
from typing import Dict, List, Any, Optional, Callable
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ResilienceConfig:
    circuit_breaker_config: CircuitBreakerConfig
    retry_config: RetryConfig
    enable_fallbacks: bool = True
    enable_circuit_breaker: bool = True
    enable_retries: bool = True
    timeout_seconds: float = 30.0

class ResilientLLMClient:
    """Cliente LLM con todos los patrones de resilience integrados"""
    
    def __init__(self, 
                 llm_client,
                 cache_client = None,
                 config: ResilienceConfig = None):
        
        self.llm_client = llm_client
        self.cache_client = cache_client
        self.config = config or ResilienceConfig(
            circuit_breaker_config=CircuitBreakerConfig(),
            retry_config=RetryConfig()
        )
        
        # Inicializar componentes
        self.circuit_breaker = AdvancedCircuitBreaker(
            "llm_service", 
            self.config.circuit_breaker_config
        ) if self.config.enable_circuit_breaker else None
        
        self.retry_manager = IntelligentRetryManager(
            self.config.retry_config
        ) if self.config.enable_retries else None
        
        self.fallback_manager = IntelligentFallbackManager() if self.config.enable_fallbacks else None
        
        # Configurar fallbacks
        if self.fallback_manager:
            self._setup_fallback_strategies()
        
        self.logger = logging.getLogger("resilient.llm.client")
    
    def _setup_fallback_strategies(self):
        """Configurar estrategias de fallback"""
        
        if self.cache_client:
            self.fallback_manager.add_strategy(
                CachedResponseFallback(self.cache_client)
            )
        
        # Añadir otras estrategias
        alternative_models = ["gpt-3.5-turbo", "text-davinci-003", "text-curie-001"]
        self.fallback_manager.add_strategy(
            AlternativeModelFallback(alternative_models, self.llm_client)
        )
        
        self.fallback_manager.add_strategy(
            SimplifiedPromptFallback(self.llm_client)
        )
        
        self.fallback_manager.add_strategy(
            StaticResponseFallback()
        )
    
    async def generate(self, 
                      prompt: str,
                      model: str = "gpt-3.5-turbo",
                      max_tokens: int = 150,
                      temperature: float = 0.7,
                      user_id: Optional[str] = None) -> Dict[str, Any]:
        """Generar texto con resilience completa"""
        
        context = {
            'original_prompt': prompt,
            'model_requested': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'user_id': user_id
        }
        
        async def llm_call():
            """Función interna para llamada LLM"""
            return await self.llm_client.generate(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        async def fallback_function():
            """Función de fallback"""
            if self.fallback_manager:
                return await self.fallback_manager.execute_fallback(context)
            else:
                raise Exception("No fallback available")
        
        try:
            # Aplicar timeout global
            if self.config.enable_circuit_breaker and self.circuit_breaker:
                # Usar circuit breaker
                result = await self.circuit_breaker.call(
                    llm_call,
                    fallback=fallback_function
                )
            elif self.config.enable_retries and self.retry_manager:
                # Usar solo retry
                result = await self.retry_manager.execute_with_retry(
                    llm_call,
                    context=context
                )
            else:
                # Llamada directa con timeout
                result = await asyncio.wait_for(
                    llm_call(),
                    timeout=self.config.timeout_seconds
                )
            
            # Enriquecer respuesta con metadatos de resilience
            if isinstance(result, dict):
                result['resilience_metadata'] = {
                    'circuit_breaker_used': self.config.enable_circuit_breaker,
                    'retry_used': self.config.enable_retries,
                    'fallback_used': result.get('fallback_type') is not None,
                    'timestamp': datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"All resilience mechanisms failed: {e}")
            
            # Último recurso: fallback estático
            if self.fallback_manager:
                try:
                    return await self.fallback_manager.execute_fallback(context)
                except Exception as fallback_error:
                    self.logger.error(f"Even fallback failed: {fallback_error}")
            
            raise e
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtener estado de salud del cliente resiliente"""
        
        health = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Estado del circuit breaker
        if self.circuit_breaker:
            cb_metrics = self.circuit_breaker.get_metrics()
            health['components']['circuit_breaker'] = {
                'status': 'healthy' if cb_metrics['state'] == 'closed' else 'degraded',
                'state': cb_metrics['state'],
                'failure_rate': cb_metrics['failure_rate'],
                'total_calls': cb_metrics['total_calls']
            }
            
            if cb_metrics['state'] != 'closed':
                health['overall_status'] = 'degraded'
        
        # Estado del retry manager
        if self.retry_manager:
            retry_metrics = self.retry_manager.get_retry_metrics()
            health['components']['retry_manager'] = {
                'status': 'healthy',
                'success_rate': retry_metrics['retry_success_rate'],
                'total_attempts': retry_metrics['total_attempts']
            }
        
        # Estado del fallback manager
        if self.fallback_manager:
            fallback_metrics = self.fallback_manager.get_fallback_metrics()
            health['components']['fallback_manager'] = {
                'status': 'healthy',
                'total_strategies': fallback_metrics['total_strategies'],
                'strategy_stats': fallback_metrics['strategy_stats']
            }
        
        return health
    
    def reset_resilience_state(self):
        """Reset de todos los componentes de resilience"""
        
        if self.circuit_breaker:
            self.circuit_breaker.reset()
        
        # Reset de métricas (si fuera necesario)
        self.logger.info("Resilience state reset completed")

# Ejemplo de uso integrado
async def example_usage():
    """Ejemplo de uso del cliente resiliente"""
    
    # Configuración personalizada
    config = ResilienceConfig(
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30
        ),
        retry_config=RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        ),
        enable_fallbacks=True,
        enable_circuit_breaker=True,
        enable_retries=True
    )
    
    # Cliente simulado (en producción sería tu cliente LLM real)
    class MockLLMClient:
        async def generate(self, **kwargs):
            # Simular fallo ocasional
            if random.random() < 0.3:
                raise RetryableError("Simulated API failure")
            
            return {
                'text': f"Generated response for: {kwargs['prompt'][:50]}...",
                'tokens_used': random.randint(20, 100)
            }
    
    # Crear cliente resiliente
    llm_client = MockLLMClient()
    resilient_client = ResilientLLMClient(
        llm_client=llm_client,
        config=config
    )
    
    # Usar el cliente
    try:
        response = await resilient_client.generate(
            prompt="Write a short story about artificial intelligence",
            model="gpt-3.5-turbo",
            max_tokens=200
        )
        
        print("Response:", response)
        
        # Ver estado de salud
        health = resilient_client.get_health_status()
        print("Health Status:", health)
        
    except Exception as e:
        print(f"Failed even with resilience: {e}")

if __name__ == "__main__":
    import random
    asyncio.run(example_usage())
```

## 🎯 Mejores Prácticas

### 1. **Circuit Breaker Configuration**
- Ajustar thresholds basados en SLAs
- Usar ventanas deslizantes para decisiones
- Implementar half-open state inteligente
- Monitorear métricas continuamente

### 2. **Fallback Design**
- Múltiples niveles de fallback
- Fallbacks específicos por tipo de request
- Cache inteligente para respuestas similares
- Respuestas estáticas como último recurso

### 3. **Retry Strategy**
- Exponential backoff con jitter
- Límites máximos de tiempo y intentos
- Diferenciación entre errores retryables y no retryables
- Retry adaptivo basado en condiciones del sistema

### 4. **Monitoring y Observabilidad**
- Métricas detalladas de cada patrón
- Alertas proactivas en degradación
- Dashboards de salud en tiempo real
- Logs estructurados para debugging

## 📊 Métricas Clave

- **Circuit Breaker State**: Closed/Open/Half-Open
- **Failure Rate**: Porcentaje de fallos
- **Recovery Time**: Tiempo de recuperación
- **Fallback Usage**: Frecuencia de fallbacks
- **Retry Success Rate**: Efectividad de reintentos

## 🔧 Próximo Paso

En la **Lección 2** implementaremos Health Checks avanzados y Service Discovery para completar el sistema de resilience.

## 📖 Recursos Adicionales

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Retry Pattern Best Practices](https://docs.microsoft.com/en-us/azure/architecture/patterns/retry)
- [Bulkhead Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/bulkhead)
- [Resilience Engineering](https://resilience-engineering.org/)
