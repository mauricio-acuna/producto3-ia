# 🧪 Laboratorio 1: Instrumentar un Agente Simple con OpenTelemetry

## 🎯 Objetivos del Laboratorio

- Configurar OpenTelemetry desde cero
- Instrumentar un agente LLM básico
- Capturar traces y métricas personalizadas
- Visualizar telemetría en Jaeger
- Enviar métricas a Prometheus

## ⏱️ Tiempo Estimado: 45 minutos

## 📋 Prerrequisitos

- Python 3.9+
- Docker y Docker Compose
- API key de OpenAI
- Editor de código (VS Code recomendado)

## 🚀 Paso 1: Configuración del Entorno

### 1.1 Crear el proyecto

```bash
mkdir lab1-otel-agent
cd lab1-otel-agent
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 1.2 Instalar dependencias

Crear `requirements.txt`:

```txt
opentelemetry-api==1.24.0
opentelemetry-sdk==1.24.0
opentelemetry-exporter-otlp==1.24.0
opentelemetry-instrumentation-requests==0.45b0
opentelemetry-instrumentation-logging==0.45b0
openai==1.35.0
python-dotenv==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
```

```bash
pip install -r requirements.txt
```

### 1.3 Variables de entorno

Crear `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key-here
OTEL_SERVICE_NAME=lab1-llm-agent
OTEL_SERVICE_VERSION=1.0.0
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
ENVIRONMENT=development
```

## 🐳 Paso 2: Configurar Infraestructura de Observabilidad

### 2.1 Docker Compose

Crear `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Jaeger - Para visualizar traces
  jaeger:
    image: jaegertracing/all-in-one:1.50
    ports:
      - "16686:16686"  # Jaeger UI
      - "14250:14250"  # gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - observability

  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.89.0
    command: ["--config=/etc/otel-collector-config.yml"]
    volumes:
      - ./config/otel-collector-config.yml:/etc/otel-collector-config.yml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8889:8889"   # Prometheus metrics
    depends_on:
      - jaeger
      - prometheus
    networks:
      - observability

  # Prometheus - Para métricas
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - observability

  # Grafana - Para dashboards
  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    networks:
      - observability

volumes:
  grafana-storage:

networks:
  observability:
    driver: bridge
```

### 2.2 Configuración del Collector

Crear `config/otel-collector-config.yml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  
  resource:
    attributes:
      - key: environment
        value: development
        action: insert

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  prometheus:
    endpoint: "0.0.0.0:8889"
    
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [jaeger, logging]
    
    metrics:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [prometheus, logging]
```

### 2.3 Configuración de Prometheus

Crear `config/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']
    scrape_interval: 5s
    metrics_path: /metrics
```

### 2.4 Levantar la infraestructura

```bash
docker-compose up -d
```

Verificar que los servicios estén funcionando:
- Jaeger UI: http://localhost:16686
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

## 💻 Paso 3: Implementar el Agente Instrumentado

### 3.1 Configuración de OpenTelemetry

Crear `src/observability.py`:

```python
import os
import logging
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.logging import LoggingInstrumentor

def setup_observability():
    """Configurar OpenTelemetry para la aplicación"""
    
    # Configurar recurso
    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "lab1-llm-agent"),
        "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
        "deployment.environment": os.getenv("ENVIRONMENT", "development")
    })
    
    # Configurar trazas
    trace_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(trace_provider)
    
    otlp_trace_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True
    )
    
    trace_provider.add_span_processor(
        BatchSpanProcessor(otlp_trace_exporter)
    )
    
    # Configurar métricas
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            insecure=True
        ),
        export_interval_millis=5000
    )
    
    metrics.set_meter_provider(MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    ))
    
    # Instrumentar logging
    LoggingInstrumentor().instrument(set_logging_format=True)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Inicializar
tracer, meter = setup_observability()
```

### 3.2 Cliente LLM Instrumentado

Crear `src/llm_client.py`:

```python
import time
import logging
from typing import Dict, List, Any
import openai
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from .observability import tracer, meter

logger = logging.getLogger(__name__)

class InstrumentedLLMClient:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
        # Métricas personalizadas
        self.request_counter = meter.create_counter(
            name="llm_requests_total",
            description="Total number of LLM requests",
            unit="1"
        )
        
        self.request_duration = meter.create_histogram(
            name="llm_request_duration_seconds",
            description="Duration of LLM requests in seconds",
            unit="s"
        )
        
        self.token_usage_counter = meter.create_counter(
            name="llm_tokens_used_total",
            description="Total tokens used in LLM requests",
            unit="1"
        )
        
        self.cost_counter = meter.create_counter(
            name="llm_cost_usd_total",
            description="Total cost in USD for LLM requests",
            unit="USD"
        )
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calcular coste aproximado basado en precios de OpenAI"""
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}
        }
        
        if model not in pricing:
            return 0.0
        
        input_cost = (prompt_tokens / 1000) * pricing[model]["input"]
        output_cost = (completion_tokens / 1000) * pricing[model]["output"]
        
        return input_cost + output_cost
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> Dict[str, Any]:
        """Realizar chat completion con instrumentación completa"""
        
        with tracer.start_as_current_span("llm.chat_completion") as span:
            start_time = time.time()
            
            # Atributos del span
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.provider", "openai")
            span.set_attribute("llm.messages.count", len(messages))
            span.set_attribute("llm.request.temperature", kwargs.get("temperature", 1.0))
            span.set_attribute("llm.request.max_tokens", kwargs.get("max_tokens", "auto"))
            
            # Log del inicio
            logger.info(f"Starting LLM request to {model} with {len(messages)} messages")
            
            try:
                # Realizar la llamada
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                
                # Calcular métricas
                duration = time.time() - start_time
                
                # Extraer información de la respuesta
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    
                    # Calcular coste
                    cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
                    
                    # Actualizar atributos del span
                    span.set_attribute("llm.usage.prompt_tokens", prompt_tokens)
                    span.set_attribute("llm.usage.completion_tokens", completion_tokens)
                    span.set_attribute("llm.usage.total_tokens", total_tokens)
                    span.set_attribute("llm.cost.usd", cost)
                else:
                    prompt_tokens = completion_tokens = total_tokens = 0
                    cost = 0.0
                
                span.set_attribute("llm.response.finish_reason", response.choices[0].finish_reason)
                span.set_attribute("llm.duration_seconds", duration)
                
                # Registrar métricas
                labels = {"model": model, "provider": "openai", "status": "success"}
                
                self.request_counter.add(1, labels)
                self.request_duration.record(duration, labels)
                
                if total_tokens > 0:
                    self.token_usage_counter.add(total_tokens, {**labels, "type": "total"})
                    self.token_usage_counter.add(prompt_tokens, {**labels, "type": "prompt"})
                    self.token_usage_counter.add(completion_tokens, {**labels, "type": "completion"})
                
                if cost > 0:
                    self.cost_counter.add(cost, labels)
                
                # Marcar éxito
                span.set_status(Status(StatusCode.OK))
                
                logger.info(
                    f"LLM request completed successfully. "
                    f"Duration: {duration:.2f}s, Tokens: {total_tokens}, Cost: ${cost:.4f}"
                )
                
                return {
                    "response": response,
                    "metrics": {
                        "duration_seconds": duration,
                        "tokens_used": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_usd": cost,
                        "model": model
                    }
                }
                
            except Exception as e:
                # Manejo de errores
                duration = time.time() - start_time
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                span.set_attribute("llm.duration_seconds", duration)
                
                # Métricas de error
                error_labels = {"model": model, "provider": "openai", "status": "error"}
                self.request_counter.add(1, error_labels)
                self.request_duration.record(duration, error_labels)
                
                logger.error(f"LLM request failed after {duration:.2f}s: {e}")
                
                raise
```

### 3.3 Agente Simple

Crear `src/agent.py`:

```python
import logging
from typing import List, Dict, Any
from opentelemetry import trace
from .llm_client import InstrumentedLLMClient
from .observability import tracer

logger = logging.getLogger(__name__)

class SimpleAgent:
    def __init__(self, llm_client: InstrumentedLLMClient):
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, str]] = []
        
        # Métricas del agente
        from .observability import meter
        self.conversation_counter = meter.create_counter(
            name="agent_conversations_total",
            description="Total number of agent conversations",
            unit="1"
        )
        
        self.message_counter = meter.create_counter(
            name="agent_messages_total",
            description="Total number of messages processed",
            unit="1"
        )
    
    def process_message(self, user_message: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Procesar un mensaje del usuario"""
        
        with tracer.start_as_current_span("agent.process_message") as span:
            span.set_attribute("agent.message.length", len(user_message))
            span.set_attribute("agent.conversation.turn", len(self.conversation_history) // 2 + 1)
            
            logger.info(f"Processing user message of length {len(user_message)}")
            
            try:
                # Agregar mensaje del usuario al historial
                self.conversation_history.append({
                    "role": "user", 
                    "content": user_message
                })
                
                # Preparar mensajes para el LLM
                messages = [
                    {"role": "system", "content": "Eres un asistente útil y amigable."}
                ] + self.conversation_history
                
                # Llamar al LLM
                result = self.llm_client.chat_completion(messages, model=model)
                
                # Extraer respuesta
                assistant_message = result["response"].choices[0].message.content
                
                # Agregar respuesta al historial
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Actualizar métricas
                self.message_counter.add(1, {"type": "user"})
                self.message_counter.add(1, {"type": "assistant"})
                
                span.set_attribute("agent.response.length", len(assistant_message))
                span.set_attribute("agent.conversation.total_messages", len(self.conversation_history))
                
                logger.info(f"Generated response of length {len(assistant_message)}")
                
                return {
                    "response": assistant_message,
                    "metrics": result["metrics"],
                    "conversation_turn": len(self.conversation_history) // 2
                }
                
            except Exception as e:
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                logger.error(f"Error processing message: {e}")
                raise
    
    def start_new_conversation(self):
        """Iniciar una nueva conversación"""
        with tracer.start_as_current_span("agent.start_conversation"):
            self.conversation_history.clear()
            self.conversation_counter.add(1, {"action": "start"})
            logger.info("Started new conversation")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Obtener resumen de la conversación actual"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m["role"] == "user"]),
            "assistant_messages": len([m for m in self.conversation_history if m["role"] == "assistant"]),
            "conversation_turns": len(self.conversation_history) // 2
        }
```

### 3.4 API Principal

Crear `src/main.py`:

```python
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from .agent import SimpleAgent
from .llm_client import InstrumentedLLMClient
from .observability import tracer

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Lab 1 - Instrumented LLM Agent", version="1.0.0")

# Inicializar agente
llm_client = InstrumentedLLMClient(api_key=os.getenv("OPENAI_API_KEY"))
agent = SimpleAgent(llm_client)

class MessageRequest(BaseModel):
    message: str
    model: Optional[str] = "gpt-3.5-turbo"

class MessageResponse(BaseModel):
    response: str
    metrics: dict
    conversation_turn: int

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Endpoint para chatear con el agente"""
    with tracer.start_as_current_span("api.chat") as span:
        span.set_attribute("http.route", "/chat")
        span.set_attribute("request.model", request.model)
        
        try:
            result = agent.process_message(request.message, request.model)
            return MessageResponse(**result)
        except Exception as e:
            span.set_attribute("error.message", str(e))
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/new-conversation")
async def new_conversation():
    """Iniciar una nueva conversación"""
    with tracer.start_as_current_span("api.new_conversation"):
        agent.start_new_conversation()
        return {"message": "New conversation started"}

@app.get("/conversation-summary")
async def conversation_summary():
    """Obtener resumen de la conversación actual"""
    return agent.get_conversation_summary()

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "service": "lab1-llm-agent"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
```

## 🧪 Paso 4: Probar la Instrumentación

### 4.1 Ejecutar la aplicación

```bash
python -m src.main
```

### 4.2 Probar los endpoints

```bash
# Nuevo conversación
curl -X POST http://localhost:8000/new-conversation

# Enviar mensaje
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué es OpenTelemetry?", "model": "gpt-3.5-turbo"}'

# Obtener resumen
curl http://localhost:8000/conversation-summary
```

### 4.3 Verificar telemetría

1. **Jaeger UI** (http://localhost:16686):
   - Buscar servicio "lab1-llm-agent"
   - Explorar traces de las llamadas
   - Verificar spans anidados

2. **Prometheus** (http://localhost:9090):
   - Consultar métricas: `llm_requests_total`
   - Verificar: `llm_request_duration_seconds`
   - Revisar: `llm_tokens_used_total`

## ✅ Criterios de Evaluación

### Básicos (Aprobatorio):
- [ ] OpenTelemetry configurado correctamente
- [ ] Traces visibles en Jaeger con información de LLM
- [ ] Métricas aparecen en Prometheus
- [ ] API responde correctamente

### Intermedios:
- [ ] Métricas personalizadas implementadas (tokens, coste)
- [ ] Manejo de errores instrumentado
- [ ] Atributos relevantes en spans
- [ ] Logs estructurados con correlación

### Avanzados:
- [ ] Contexto distribuido funcionando
- [ ] Métricas de negocio (conversaciones, turns)
- [ ] Performance optimizada (batching, sampling)
- [ ] Documentación de métricas creada

## 🎉 ¡Felicitaciones!

Has completado exitosamente la instrumentación básica de un agente LLM con OpenTelemetry. En el próximo laboratorio aprenderemos a crear dashboards avanzados en Grafana y configurar alertas inteligentes.

## 📚 Para Profundizar

- Experimenta con diferentes modelos y compara métricas
- Agrega más métricas personalizadas
- Implementa sampling de traces para reducir overhead
- Prueba con cargas de trabajo más intensas
