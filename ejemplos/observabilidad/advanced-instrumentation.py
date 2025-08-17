# Ejemplo: Advanced OpenTelemetry Instrumentation

Este ejemplo muestra técnicas avanzadas de instrumentación para agentes LLM.

## Custom Metrics y Attributes

```python
import time
import hashlib
from typing import Dict, Any, Optional
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
from enum import Enum

class LLMMetrics:
    """Clase para gestionar métricas personalizadas de LLM"""
    
    def __init__(self, meter):
        self.meter = meter
        
        # Contadores
        self.request_counter = meter.create_counter(
            name="llm_requests_total",
            description="Total LLM requests",
            unit="1"
        )
        
        self.token_counter = meter.create_counter(
            name="llm_tokens_total", 
            description="Total tokens processed",
            unit="1"
        )
        
        self.cost_counter = meter.create_counter(
            name="llm_cost_total",
            description="Total cost in USD",
            unit="USD"
        )
        
        # Histogramas
        self.latency_histogram = meter.create_histogram(
            name="llm_request_duration_seconds",
            description="Request duration in seconds",
            unit="s"
        )
        
        self.token_efficiency_histogram = meter.create_histogram(
            name="llm_tokens_per_request",
            description="Tokens used per request",
            unit="1"
        )
        
        # Gauges para métricas instantáneas
        self.active_conversations = meter.create_up_down_counter(
            name="llm_active_conversations",
            description="Currently active conversations",
            unit="1"
        )
        
        self.queue_size = meter.create_up_down_counter(
            name="llm_request_queue_size",
            description="Current request queue size",
            unit="1"
        )

class ConversationContext:
    """Context manager para conversaciones instrumentadas"""
    
    def __init__(self, tracer, metrics: LLMMetrics, conversation_id: str):
        self.tracer = tracer
        self.metrics = metrics
        self.conversation_id = conversation_id
        self.span = None
        
    def __enter__(self):
        self.span = self.tracer.start_span("conversation.session")
        self.span.set_attribute("conversation.id", self.conversation_id)
        self.span.set_attribute("conversation.start_time", time.time())
        
        # Incrementar conversaciones activas
        self.metrics.active_conversations.add(1)
        
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.set_attribute("error.type", exc_type.__name__)
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.end()
        
        # Decrementar conversaciones activas
        self.metrics.active_conversations.add(-1)

class InstrumentedLLMProvider:
    """Provider de LLM con instrumentación avanzada"""
    
    def __init__(self, tracer, metrics: LLMMetrics, provider_name: str):
        self.tracer = tracer
        self.metrics = metrics
        self.provider_name = provider_name
        
    def _calculate_request_hash(self, messages: list, model: str) -> str:
        """Calcular hash del request para cache tracking"""
        content = f"{model}:{str(messages)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _extract_conversation_info(self, messages: list) -> Dict[str, Any]:
        """Extraer información de contexto de la conversación"""
        user_messages = [m for m in messages if m.get("role") == "user"]
        system_messages = [m for m in messages if m.get("role") == "system"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "system_messages": len(system_messages),
            "assistant_messages": len(assistant_messages),
            "conversation_turns": len(user_messages),
            "has_system_prompt": len(system_messages) > 0
        }
    
    def chat_completion(self, 
                       messages: list, 
                       model: str,
                       conversation_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """Chat completion con instrumentación completa"""
        
        with self.tracer.start_as_current_span("llm.chat_completion") as span:
            start_time = time.time()
            request_hash = self._calculate_request_hash(messages, model)
            conv_info = self._extract_conversation_info(messages)
            
            # Atributos básicos del span
            span.set_attribute("llm.provider", self.provider_name)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.request.hash", request_hash)
            
            # Información de conversación
            for key, value in conv_info.items():
                span.set_attribute(f"llm.conversation.{key}", value)
            
            # Contexto de usuario
            if conversation_id:
                span.set_attribute("llm.conversation_id", conversation_id)
            if user_id:
                span.set_attribute("llm.user_id", user_id)
            
            # Parámetros del modelo
            span.set_attribute("llm.temperature", kwargs.get("temperature", 1.0))
            span.set_attribute("llm.max_tokens", kwargs.get("max_tokens", "auto"))
            
            try:
                # Simular llamada al LLM (aquí iría la llamada real)
                # response = await self.client.chat.completions.create(...)
                
                # Para el ejemplo, simular respuesta
                response = self._simulate_response(messages, model)
                
                # Métricas post-respuesta
                duration = time.time() - start_time
                
                # Información de uso de tokens
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # Calcular coste
                cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
                
                # Actualizar atributos del span
                span.set_attribute("llm.usage.prompt_tokens", prompt_tokens)
                span.set_attribute("llm.usage.completion_tokens", completion_tokens) 
                span.set_attribute("llm.usage.total_tokens", total_tokens)
                span.set_attribute("llm.cost_usd", cost)
                span.set_attribute("llm.duration_seconds", duration)
                span.set_attribute("llm.tokens_per_second", total_tokens / duration if duration > 0 else 0)
                
                # Métricas
                labels = {
                    "model": model,
                    "provider": self.provider_name,
                    "status": "success",
                    "conversation_turns": str(conv_info["conversation_turns"])
                }
                
                if user_id:
                    labels["user_id_hash"] = hashlib.md5(user_id.encode()).hexdigest()[:8]
                
                self.metrics.request_counter.add(1, labels)
                self.metrics.latency_histogram.record(duration, labels)
                self.metrics.token_counter.add(total_tokens, {**labels, "type": "total"})
                self.metrics.token_counter.add(prompt_tokens, {**labels, "type": "prompt"})
                self.metrics.token_counter.add(completion_tokens, {**labels, "type": "completion"})
                self.metrics.cost_counter.add(cost, labels)
                self.metrics.token_efficiency_histogram.record(total_tokens, labels)
                
                # Métricas de calidad (ejemplo)
                quality_score = self._calculate_quality_score(response)
                span.set_attribute("llm.quality_score", quality_score)
                
                span.set_status(Status(StatusCode.OK))
                
                return {
                    "response": response,
                    "metrics": {
                        "duration": duration,
                        "tokens": total_tokens,
                        "cost": cost,
                        "quality_score": quality_score,
                        "request_hash": request_hash
                    }
                }
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Instrumentar error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                # Métricas de error
                error_labels = {
                    "model": model,
                    "provider": self.provider_name,
                    "status": "error",
                    "error_type": type(e).__name__
                }
                
                self.metrics.request_counter.add(1, error_labels)
                self.metrics.latency_histogram.record(duration, error_labels)
                
                raise
    
    def _simulate_response(self, messages: list, model: str) -> dict:
        """Simular respuesta del LLM para el ejemplo"""
        import random
        
        # Simular tokens basado en el input
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages) * 1.3
        completion_tokens = random.randint(50, 200)
        total_tokens = int(prompt_tokens + completion_tokens)
        
        return {
            "choices": [{
                "message": {
                    "content": f"Esta es una respuesta simulada del modelo {model}.",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "model": model
        }
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calcular coste del request"""
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        }
        
        if model not in pricing:
            return 0.0
        
        input_cost = (prompt_tokens / 1000) * pricing[model]["input"]
        output_cost = (completion_tokens / 1000) * pricing[model]["output"]
        
        return input_cost + output_cost
    
    def _calculate_quality_score(self, response: dict) -> float:
        """Calcular score de calidad de la respuesta (ejemplo simple)"""
        import random
        
        # En un caso real, esto podría usar métricas como:
        # - Longitud apropiada de respuesta
        # - Relevancia semántica
        # - Coherencia
        # - Safety score
        
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Score básico basado en longitud y estructura
        length_score = min(len(content) / 500, 1.0)  # Normalizar por longitud
        structure_score = 0.8 if len(content.split('.')) > 1 else 0.5  # Tiene múltiples oraciones
        
        # Añadir algo de randomness para simular variabilidad real
        noise = random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, (length_score + structure_score) / 2 + noise))

# Ejemplo de uso
if __name__ == "__main__":
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    
    # Setup básico
    trace.set_tracer_provider(TracerProvider())
    metrics.set_meter_provider(MeterProvider())
    
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    # Crear instancias
    llm_metrics = LLMMetrics(meter)
    provider = InstrumentedLLMProvider(tracer, llm_metrics, "openai")
    
    # Simular conversación
    conversation_id = "conv_123"
    user_id = "user_456"
    
    with ConversationContext(tracer, llm_metrics, conversation_id):
        messages = [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": "¿Qué es OpenTelemetry?"}
        ]
        
        result = provider.chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",
            conversation_id=conversation_id,
            user_id=user_id,
            temperature=0.7
        )
        
        print(f"Response: {result['response']['choices'][0]['message']['content']}")
        print(f"Metrics: {result['metrics']}")
```

## Métricas Avanzadas

### 1. Business Metrics

```python
class BusinessMetrics:
    """Métricas de negocio para LLMOps"""
    
    def __init__(self, meter):
        # User engagement
        self.user_sessions = meter.create_counter(
            name="user_sessions_total",
            description="Total user sessions"
        )
        
        self.session_duration = meter.create_histogram(
            name="user_session_duration_seconds", 
            description="User session duration"
        )
        
        # Content quality
        self.user_satisfaction = meter.create_histogram(
            name="user_satisfaction_score",
            description="User satisfaction score (1-5)"
        )
        
        # Revenue impact
        self.revenue_per_user = meter.create_histogram(
            name="revenue_per_user_usd",
            description="Revenue generated per user"
        )
```

### 2. Performance Metrics

```python
class PerformanceMetrics:
    """Métricas de rendimiento detalladas"""
    
    def __init__(self, meter):
        # Throughput
        self.requests_per_second = meter.create_histogram(
            name="requests_per_second",
            description="Current RPS"
        )
        
        # Queue metrics
        self.queue_wait_time = meter.create_histogram(
            name="queue_wait_time_seconds",
            description="Time spent waiting in queue"
        )
        
        # Cache performance
        self.cache_hit_ratio = meter.create_histogram(
            name="cache_hit_ratio",
            description="Cache hit ratio"
        )
        
        # Model switching
        self.model_fallbacks = meter.create_counter(
            name="model_fallbacks_total",
            description="Number of model fallbacks"
        )
```

Este ejemplo muestra técnicas avanzadas de instrumentación que van más allá de las métricas básicas, incluyendo contexto de conversación, métricas de negocio y análisis de calidad de respuestas.
