# 🧪 Laboratorio 6: Load Balancing y Auto-scaling

## 🎯 Objetivos del Laboratorio

En este laboratorio implementaremos:
- Load balancer inteligente con múltiples estrategias
- Rate limiting avanzado con Redis
- Auto-scaling basado en métricas
- Monitoreo de performance en tiempo real
- Dashboard de observabilidad

## 🏗️ Arquitectura del Sistema

```mermaid
graph TB
    A[Client] --> B[Load Balancer]
    B --> C[Rate Limiter]
    C --> D[API Gateway]
    D --> E[LLM Instance 1]
    D --> F[LLM Instance 2]
    D --> G[LLM Instance N]
    
    H[Auto Scaler] --> I[Metrics Collector]
    I --> E
    I --> F
    I --> G
    
    J[Redis Cache] --> C
    K[Prometheus] --> I
    L[Grafana] --> K
```

## 🚀 Implementación Completa

### 1. Docker Compose para la Infraestructura

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Redis para caching y rate limiting
  redis:
    image: redis:7-alpine
    container_name: llm_redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - llm_network

  # PostgreSQL para métricas persistentes
  postgres:
    image: postgres:15
    container_name: llm_postgres
    environment:
      POSTGRES_DB: llm_metrics
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - llm_network

  # Prometheus para métricas
  prometheus:
    image: prom/prometheus:latest
    container_name: llm_prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - llm_network

  # Grafana para dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: llm_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - llm_network

  # Load Balancer Principal
  load_balancer:
    build:
      context: .
      dockerfile: Dockerfile.loadbalancer
    container_name: llm_load_balancer
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://admin:admin123@postgres:5432/llm_metrics
    depends_on:
      - redis
      - postgres
    networks:
      - llm_network

  # LLM Instance 1
  llm_instance_1:
    build:
      context: .
      dockerfile: Dockerfile.llm
    container_name: llm_instance_1
    ports:
      - "8001:8000"
    environment:
      - INSTANCE_ID=instance_1
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - llm_network

  # LLM Instance 2
  llm_instance_2:
    build:
      context: .
      dockerfile: Dockerfile.llm
    container_name: llm_instance_2
    ports:
      - "8002:8000"
    environment:
      - INSTANCE_ID=instance_2
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - llm_network

  # Auto Scaler
  auto_scaler:
    build:
      context: .
      dockerfile: Dockerfile.autoscaler
    container_name: llm_auto_scaler
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - prometheus
      - load_balancer
    networks:
      - llm_network

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  llm_network:
    driver: bridge
```

### 2. Load Balancer con FastAPI

```python
# load_balancer/main.py
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import time
import uuid
import json
import logging
from datetime import datetime

import redis.asyncio as redis
import asyncpg
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Importar nuestras clases personalizadas
from performance_monitor import PerformanceMonitor, MetricType
from load_balancer_core import IntelligentLoadBalancer, BackendServer, LoadBalancingStrategy
from rate_limiter import AdvancedRateLimiter, RateLimitStrategy

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas de Prometheus
REQUEST_COUNT = Counter('llm_requests_total', 'Total LLM requests', ['backend', 'status'])
REQUEST_DURATION = Histogram('llm_request_duration_seconds', 'Request duration', ['backend'])
ACTIVE_CONNECTIONS = Gauge('llm_active_connections', 'Active connections', ['backend'])
RATE_LIMIT_HITS = Counter('llm_rate_limit_hits_total', 'Rate limit hits', ['identifier_type'])

app = FastAPI(title="LLM Load Balancer", version="1.0.0")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class LLMRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.7
    user_id: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    backend_id: str
    processing_time_ms: float
    tokens_used: int
    request_id: str

# Variables globales
redis_client = None
db_pool = None
load_balancer = None
rate_limiter = None
performance_monitor = None

@app.on_event("startup")
async def startup_event():
    """Inicializar servicios"""
    global redis_client, db_pool, load_balancer, rate_limiter, performance_monitor
    
    # Conectar Redis
    redis_client = redis.from_url("redis://redis:6379", decode_responses=True)
    
    # Conectar PostgreSQL
    db_pool = await asyncpg.create_pool(
        "postgresql://admin:admin123@postgres:5432/llm_metrics",
        min_size=5,
        max_size=20
    )
    
    # Inicializar componentes
    performance_monitor = PerformanceMonitor()
    rate_limiter = AdvancedRateLimiter(redis_client)
    
    # Crear load balancer
    load_balancer = IntelligentLoadBalancer(LoadBalancingStrategy.ADAPTIVE)
    
    # Registrar backends
    backends = [
        BackendServer(id="instance_1", url="http://llm_instance_1:8000", weight=1.0),
        BackendServer(id="instance_2", url="http://llm_instance_2:8000", weight=1.0),
    ]
    
    for backend in backends:
        load_balancer.add_backend(backend)
    
    # Iniciar health monitoring
    await load_balancer.start_health_monitoring()
    
    logger.info("Load balancer started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup al cerrar"""
    if load_balancer:
        load_balancer.stop_health_monitoring()
    
    if db_pool:
        await db_pool.close()
    
    if redis_client:
        await redis_client.close()

async def get_client_identifier(request: Request) -> str:
    """Obtener identificador del cliente"""
    # Prioridad: API key > User ID > IP
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"
    
    client_ip = request.client.host
    return f"ip:{client_ip}"

async def make_llm_request(backend_url: str, llm_request: LLMRequest) -> Dict[str, Any]:
    """Hacer request al backend LLM"""
    
    async with aiohttp.ClientSession() as session:
        payload = {
            "prompt": llm_request.prompt,
            "model": llm_request.model,
            "max_tokens": llm_request.max_tokens,
            "temperature": llm_request.temperature
        }
        
        async with session.post(
            f"{backend_url}/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Backend error: {response.status}")

@app.post("/generate", response_model=LLMResponse)
async def generate_text(
    llm_request: LLMRequest, 
    request: Request,
    background_tasks: BackgroundTasks
):
    """Endpoint principal para generación de texto"""
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Obtener identificador del cliente
        client_id = await get_client_identifier(request)
        
        # Verificar rate limit
        rate_limit_result = await rate_limiter.check_rate_limit(
            identifier=client_id,
            rule_type='per_user'
        )
        
        if not rate_limit_result['allowed']:
            RATE_LIMIT_HITS.labels(identifier_type=client_id.split(':')[0]).inc()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": rate_limit_result.get('retry_after_seconds', 60),
                    "limit_info": rate_limit_result
                }
            )
        
        # Ejecutar request con load balancing
        result = await load_balancer.execute_request(
            request_func=make_llm_request,
            llm_request,
            max_retries=2,
            model_name=llm_request.model,
            input_tokens=len(llm_request.prompt.split()),  # Aproximación simple
            context={'user_id': llm_request.user_id}
        )
        
        if not result['success']:
            REQUEST_COUNT.labels(backend='unknown', status='error').inc()
            raise HTTPException(
                status_code=503,
                detail=f"Service unavailable: {result.get('error', 'Unknown error')}"
            )
        
        # Extraer información de la respuesta
        backend_id = result['backend_id']
        processing_time = result['response_time_ms']
        llm_response = result['result']
        
        # Actualizar métricas de Prometheus
        REQUEST_COUNT.labels(backend=backend_id, status='success').inc()
        REQUEST_DURATION.labels(backend=backend_id).observe(processing_time / 1000)
        
        # Actualizar success rate para rate limiting adaptivo
        background_tasks.add_task(
            rate_limiter.update_success_rate,
            client_id,
            True
        )
        
        # Persistir métricas en base de datos
        background_tasks.add_task(
            save_request_metrics,
            request_id,
            client_id,
            backend_id,
            llm_request.model,
            processing_time,
            len(llm_request.prompt.split()),
            llm_response.get('tokens_used', 0),
            True
        )
        
        return LLMResponse(
            response=llm_response.get('text', ''),
            backend_id=backend_id,
            processing_time_ms=processing_time,
            tokens_used=llm_response.get('tokens_used', 0),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        
        # Actualizar métricas de error
        REQUEST_COUNT.labels(backend='unknown', status='error').inc()
        
        # Actualizar success rate
        client_id = await get_client_identifier(request)
        background_tasks.add_task(
            rate_limiter.update_success_rate,
            client_id,
            False
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

async def save_request_metrics(
    request_id: str,
    client_id: str,
    backend_id: str,
    model: str,
    processing_time: float,
    input_tokens: int,
    output_tokens: int,
    success: bool
):
    """Guardar métricas en base de datos"""
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO request_metrics (
                request_id, client_id, backend_id, model, 
                processing_time_ms, input_tokens, output_tokens, 
                success, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """, request_id, client_id, backend_id, model, 
             processing_time, input_tokens, output_tokens, success, datetime.now())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "load_balancer": load_balancer.get_backend_status() if load_balancer else "not_initialized"
    }

@app.get("/metrics")
async def get_prometheus_metrics():
    """Endpoint para métricas de Prometheus"""
    return generate_latest()

@app.get("/status")
async def get_system_status():
    """Estado completo del sistema"""
    
    performance_summary = performance_monitor.get_performance_summary(5)
    backend_status = load_balancer.get_backend_status()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "performance_summary": performance_summary,
        "backend_status": backend_status,
        "active_backends": backend_status['healthy_backends'],
        "total_backends": backend_status['total_backends'],
        "system_health": "healthy" if backend_status['healthy_backends'] > 0 else "degraded"
    }

@app.get("/rate-limits/{identifier}")
async def get_rate_limit_status(identifier: str):
    """Obtener estado de rate limits para un identificador"""
    
    stats = await rate_limiter.get_rate_limit_stats(identifier)
    return stats

@app.post("/admin/reset-rate-limit/{identifier}")
async def reset_rate_limit(identifier: str, rule_type: Optional[str] = None):
    """Reset rate limit para un identificador (admin only)"""
    
    await rate_limiter.reset_rate_limit(identifier, rule_type)
    return {"message": f"Rate limit reset for {identifier}"}

@app.post("/admin/add-backend")
async def add_backend(backend_data: Dict[str, Any]):
    """Añadir nuevo backend (admin only)"""
    
    backend = BackendServer(
        id=backend_data['id'],
        url=backend_data['url'],
        weight=backend_data.get('weight', 1.0),
        max_connections=backend_data.get('max_connections', 100)
    )
    
    load_balancer.add_backend(backend)
    
    return {"message": f"Backend {backend.id} added successfully"}

@app.delete("/admin/remove-backend/{backend_id}")
async def remove_backend(backend_id: str):
    """Remover backend (admin only)"""
    
    load_balancer.remove_backend(backend_id)
    
    return {"message": f"Backend {backend_id} removed successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. LLM Instance (Backend)

```python
# llm_instance/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
import os
import logging
import asyncio
import random

import openai
from prometheus_client import Counter, Histogram, generate_latest

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Métricas
REQUESTS_TOTAL = Counter('llm_instance_requests_total', 'Total requests to this instance')
REQUEST_DURATION = Histogram('llm_instance_duration_seconds', 'Request duration')
TOKENS_GENERATED = Counter('llm_instance_tokens_total', 'Total tokens generated')

app = FastAPI(title=f"LLM Instance - {os.getenv('INSTANCE_ID', 'unknown')}")

instance_id = os.getenv('INSTANCE_ID', 'unknown')

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    tokens_used: int
    model: str
    instance_id: str
    processing_time_ms: float

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generar texto usando LLM"""
    
    start_time = time.time()
    
    try:
        REQUESTS_TOTAL.inc()
        
        # Simular latencia variable
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        if openai.api_key:
            # Usar OpenAI real
            response = await openai.ChatCompletion.acreate(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
        else:
            # Simulación para testing
            text = f"Simulated response to: {request.prompt[:50]}..."
            tokens_used = random.randint(50, request.max_tokens)
            
            # Simular processing time
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        processing_time = (time.time() - start_time) * 1000
        
        # Actualizar métricas
        REQUEST_DURATION.observe(processing_time / 1000)
        TOKENS_GENERATED.inc(tokens_used)
        
        return GenerateResponse(
            text=text,
            tokens_used=tokens_used,
            model=request.model,
            instance_id=instance_id,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "instance_id": instance_id,
        "timestamp": time.time()
    }

@app.get("/metrics")
async def metrics():
    """Métricas de Prometheus"""
    return generate_latest()

@app.get("/load")
async def simulate_load():
    """Simular carga para testing"""
    load_factor = random.uniform(0.1, 2.0)
    await asyncio.sleep(load_factor)
    
    return {
        "instance_id": instance_id,
        "simulated_load_seconds": load_factor,
        "cpu_simulation": random.uniform(20, 90),
        "memory_simulation": random.uniform(40, 85)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 4. Auto-scaler

```python
# auto_scaler/main.py
import asyncio
import time
import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

import aiohttp
import docker
from prometheus_client.parser import text_string_to_metric_families

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerAutoScaler:
    """Auto-scaler que gestiona containers Docker"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.prometheus_url = "http://prometheus:9090"
        self.load_balancer_url = "http://load_balancer:8000"
        
        # Configuración de scaling
        self.min_instances = 1
        self.max_instances = 5
        self.current_instances = 2
        self.scaling_cooldown = 300  # 5 minutos
        self.last_scaling_action = None
        
        # Thresholds
        self.cpu_threshold_up = 70.0
        self.cpu_threshold_down = 30.0
        self.memory_threshold_up = 80.0
        self.memory_threshold_down = 40.0
        self.response_time_threshold = 2.0
        
    async def get_metrics_from_prometheus(self) -> Dict[str, float]:
        """Obtener métricas de Prometheus"""
        
        queries = {
            'avg_cpu': 'rate(container_cpu_usage_seconds_total[5m]) * 100',
            'avg_memory': 'container_memory_usage_bytes / container_spec_memory_limit_bytes * 100',
            'avg_response_time': 'rate(llm_request_duration_seconds_sum[5m]) / rate(llm_request_duration_seconds_count[5m])',
            'request_rate': 'rate(llm_requests_total[5m]) * 60',
            'error_rate': 'rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) * 100'
        }
        
        metrics = {}
        
        async with aiohttp.ClientSession() as session:
            for metric_name, query in queries.items():
                try:
                    async with session.get(
                        f"{self.prometheus_url}/api/v1/query",
                        params={'query': query}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data['data']['result']:
                                value = float(data['data']['result'][0]['value'][1])
                                metrics[metric_name] = value
                            else:
                                metrics[metric_name] = 0.0
                        else:
                            logger.warning(f"Failed to get metric {metric_name}")
                            metrics[metric_name] = 0.0
                            
                except Exception as e:
                    logger.error(f"Error getting metric {metric_name}: {e}")
                    metrics[metric_name] = 0.0
        
        return metrics
    
    def should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Determinar si debe escalar hacia arriba"""
        
        conditions = [
            metrics.get('avg_cpu', 0) > self.cpu_threshold_up,
            metrics.get('avg_memory', 0) > self.memory_threshold_up,
            metrics.get('avg_response_time', 0) > self.response_time_threshold
        ]
        
        # Al menos 2 de 3 condiciones deben cumplirse
        return sum(conditions) >= 2
    
    def should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Determinar si debe escalar hacia abajo"""
        
        conditions = [
            metrics.get('avg_cpu', 100) < self.cpu_threshold_down,
            metrics.get('avg_memory', 100) < self.memory_threshold_down,
            metrics.get('avg_response_time', 10) < self.response_time_threshold / 2
        ]
        
        # Todas las condiciones deben cumplirse para scale down
        return all(conditions)
    
    def is_in_cooldown(self) -> bool:
        """Verificar si estamos en cooldown"""
        
        if not self.last_scaling_action:
            return False
        
        elapsed = (datetime.now() - self.last_scaling_action).total_seconds()
        return elapsed < self.scaling_cooldown
    
    async def scale_up(self) -> bool:
        """Escalar hacia arriba añadiendo una instancia"""
        
        if self.current_instances >= self.max_instances:
            logger.info("Maximum instances reached, cannot scale up")
            return False
        
        try:
            new_instance_id = f"llm_instance_{self.current_instances + 1}"
            port = 8000 + self.current_instances + 1
            
            # Crear nuevo container
            container = self.docker_client.containers.run(
                image="llm_instance:latest",
                name=new_instance_id,
                ports={'8000/tcp': port},
                environment={
                    'INSTANCE_ID': new_instance_id,
                    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', '')
                },
                network="producto3_llm_network",
                detach=True
            )
            
            # Esperar a que el container esté listo
            await asyncio.sleep(10)
            
            # Registrar en load balancer
            await self.register_backend_in_load_balancer(new_instance_id, port)
            
            self.current_instances += 1
            self.last_scaling_action = datetime.now()
            
            logger.info(f"Scaled up: Added {new_instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error scaling up: {e}")
            return False
    
    async def scale_down(self) -> bool:
        """Escalar hacia abajo removiendo una instancia"""
        
        if self.current_instances <= self.min_instances:
            logger.info("Minimum instances reached, cannot scale down")
            return False
        
        try:
            instance_to_remove = f"llm_instance_{self.current_instances}"
            
            # Remover del load balancer primero
            await self.unregister_backend_from_load_balancer(instance_to_remove)
            
            # Esperar a que drenen las conexiones
            await asyncio.sleep(30)
            
            # Detener y remover container
            try:
                container = self.docker_client.containers.get(instance_to_remove)
                container.stop()
                container.remove()
            except docker.errors.NotFound:
                logger.warning(f"Container {instance_to_remove} not found")
            
            self.current_instances -= 1
            self.last_scaling_action = datetime.now()
            
            logger.info(f"Scaled down: Removed {instance_to_remove}")
            return True
            
        except Exception as e:
            logger.error(f"Error scaling down: {e}")
            return False
    
    async def register_backend_in_load_balancer(self, instance_id: str, port: int):
        """Registrar nuevo backend en load balancer"""
        
        backend_data = {
            'id': instance_id,
            'url': f'http://{instance_id}:8000',
            'weight': 1.0,
            'max_connections': 100
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.load_balancer_url}/admin/add-backend",
                    json=backend_data
                ) as response:
                    if response.status == 200:
                        logger.info(f"Registered backend {instance_id} in load balancer")
                    else:
                        logger.error(f"Failed to register backend: {response.status}")
            except Exception as e:
                logger.error(f"Error registering backend: {e}")
    
    async def unregister_backend_from_load_balancer(self, instance_id: str):
        """Desregistrar backend del load balancer"""
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.delete(
                    f"{self.load_balancer_url}/admin/remove-backend/{instance_id}"
                ) as response:
                    if response.status == 200:
                        logger.info(f"Unregistered backend {instance_id} from load balancer")
                    else:
                        logger.error(f"Failed to unregister backend: {response.status}")
            except Exception as e:
                logger.error(f"Error unregistering backend: {e}")
    
    async def run_scaling_loop(self):
        """Loop principal de auto-scaling"""
        
        logger.info("Starting auto-scaling loop")
        
        while True:
            try:
                # Obtener métricas
                metrics = await self.get_metrics_from_prometheus()
                
                logger.info(f"Current metrics: {json.dumps(metrics, indent=2)}")
                logger.info(f"Current instances: {self.current_instances}")
                
                if self.is_in_cooldown():
                    logger.info("In scaling cooldown, skipping scaling decision")
                else:
                    # Evaluar scaling
                    if self.should_scale_up(metrics):
                        logger.info("Scaling up triggered")
                        success = await self.scale_up()
                        if success:
                            logger.info("Scale up completed successfully")
                        else:
                            logger.error("Scale up failed")
                    
                    elif self.should_scale_down(metrics):
                        logger.info("Scaling down triggered")
                        success = await self.scale_down()
                        if success:
                            logger.info("Scale down completed successfully")
                        else:
                            logger.error("Scale down failed")
                    
                    else:
                        logger.info("No scaling action needed")
                
                # Esperar antes del próximo ciclo
                await asyncio.sleep(60)  # Evaluar cada minuto
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(30)  # Esperar menos tiempo si hay error

if __name__ == "__main__":
    scaler = DockerAutoScaler()
    asyncio.run(scaler.run_scaling_loop())
```

### 5. Configuración de Monitoring

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'load-balancer'
    static_configs:
      - targets: ['load_balancer:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'llm-instances'
    static_configs:
      - targets: 
        - 'llm_instance_1:8000'
        - 'llm_instance_2:8000'
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

```sql
-- init.sql
CREATE DATABASE IF NOT EXISTS llm_metrics;

USE llm_metrics;

CREATE TABLE request_metrics (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    backend_id VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    processing_time_ms FLOAT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    success BOOLEAN NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_client_id (client_id),
    INDEX idx_backend_id (backend_id),
    INDEX idx_success (success)
);

CREATE TABLE scaling_events (
    id SERIAL PRIMARY KEY,
    event_type ENUM('scale_up', 'scale_down') NOT NULL,
    old_instances INTEGER NOT NULL,
    new_instances INTEGER NOT NULL,
    trigger_metrics JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6. Dockerfiles

```dockerfile
# Dockerfile.loadbalancer
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY load_balancer/ .
COPY performance_monitor.py .
COPY load_balancer_core.py .
COPY rate_limiter.py .

EXPOSE 8000

CMD ["python", "main.py"]
```

```dockerfile
# Dockerfile.llm
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY llm_instance/ .

EXPOSE 8000

CMD ["python", "main.py"]
```

```dockerfile
# Dockerfile.autoscaler
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y docker.io

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY auto_scaler/ .

CMD ["python", "main.py"]
```

### 7. Scripts de Testing

```python
# test_load_balancer.py
import asyncio
import aiohttp
import time
import random
from typing import List, Dict, Any

class LoadTester:
    """Tester para el sistema de load balancing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def make_request(self, session: aiohttp.ClientSession, user_id: str) -> Dict[str, Any]:
        """Hacer una request individual"""
        
        start_time = time.time()
        
        payload = {
            "prompt": f"Tell me a story about {random.choice(['cats', 'dogs', 'dragons', 'space', 'magic'])}",
            "model": "gpt-3.5-turbo",
            "max_tokens": 100,
            "temperature": 0.7,
            "user_id": user_id
        }
        
        headers = {
            "X-User-ID": user_id,
            "Content-Type": "application/json"
        }
        
        try:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                
                result = {
                    'user_id': user_id,
                    'status_code': response.status,
                    'response_time': (end_time - start_time) * 1000,
                    'timestamp': start_time,
                    'success': response.status == 200
                }
                
                if response.status == 200:
                    data = await response.json()
                    result.update({
                        'backend_id': data.get('backend_id'),
                        'processing_time_ms': data.get('processing_time_ms'),
                        'tokens_used': data.get('tokens_used')
                    })
                else:
                    result['error'] = await response.text()
                
                return result
                
        except Exception as e:
            return {
                'user_id': user_id,
                'status_code': 0,
                'response_time': (time.time() - start_time) * 1000,
                'timestamp': start_time,
                'success': False,
                'error': str(e)
            }
    
    async def run_load_test(self, 
                          concurrent_users: int = 10,
                          requests_per_user: int = 20,
                          delay_between_requests: float = 1.0):
        """Ejecutar test de carga"""
        
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for user_num in range(concurrent_users):
                user_id = f"test_user_{user_num}"
                user_tasks = []
                
                for req_num in range(requests_per_user):
                    task = self.make_request(session, user_id)
                    user_tasks.append(task)
                    
                    # Espaciar requests
                    if req_num < requests_per_user - 1:
                        await asyncio.sleep(delay_between_requests)
                
                tasks.extend(user_tasks)
            
            # Ejecutar todas las requests
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrar excepciones
            self.results = [r for r in results if isinstance(r, dict)]
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analizar resultados del test"""
        
        if not self.results:
            return {"error": "No results to analyze"}
        
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r['success']])
        failed_requests = total_requests - successful_requests
        
        response_times = [r['response_time'] for r in self.results if r['success']]
        
        # Análisis por backend
        backend_stats = {}
        for result in self.results:
            if result['success'] and 'backend_id' in result:
                backend_id = result['backend_id']
                if backend_id not in backend_stats:
                    backend_stats[backend_id] = {
                        'requests': 0,
                        'total_time': 0,
                        'avg_time': 0
                    }
                
                backend_stats[backend_id]['requests'] += 1
                backend_stats[backend_id]['total_time'] += result['response_time']
        
        # Calcular promedios
        for backend_id, stats in backend_stats.items():
            stats['avg_time'] = stats['total_time'] / stats['requests']
        
        analysis = {
            'summary': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate_percent': (successful_requests / total_requests) * 100,
                'error_rate_percent': (failed_requests / total_requests) * 100
            },
            'performance': {
                'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
                'min_response_time_ms': min(response_times) if response_times else 0,
                'max_response_time_ms': max(response_times) if response_times else 0,
                'p95_response_time_ms': self._percentile(response_times, 95) if response_times else 0,
                'p99_response_time_ms': self._percentile(response_times, 99) if response_times else 0
            },
            'load_balancing': {
                'backend_distribution': backend_stats,
                'load_balance_effectiveness': self._calculate_load_balance_score(backend_stats)
            }
        }
        
        return analysis
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcular percentil"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_load_balance_score(self, backend_stats: Dict[str, Dict]) -> float:
        """Calcular score de efectividad del load balancing"""
        
        if len(backend_stats) < 2:
            return 100.0  # Solo un backend
        
        request_counts = [stats['requests'] for stats in backend_stats.values()]
        avg_requests = sum(request_counts) / len(request_counts)
        
        # Calcular variación
        variance = sum((count - avg_requests) ** 2 for count in request_counts) / len(request_counts)
        std_dev = variance ** 0.5
        
        # Score basado en qué tan cerca está la distribución del ideal
        if avg_requests == 0:
            return 0.0
        
        coefficient_of_variation = std_dev / avg_requests
        balance_score = max(0, 100 - (coefficient_of_variation * 100))
        
        return round(balance_score, 2)

async def main():
    """Función principal para testing"""
    
    tester = LoadTester()
    
    print("Running load test...")
    await tester.run_load_test(
        concurrent_users=5,
        requests_per_user=10,
        delay_between_requests=0.5
    )
    
    print("Analyzing results...")
    analysis = tester.analyze_results()
    
    print("\n" + "="*50)
    print("LOAD TEST RESULTS")
    print("="*50)
    
    print(f"""
Summary:
  Total Requests: {analysis['summary']['total_requests']}
  Successful: {analysis['summary']['successful_requests']}
  Failed: {analysis['summary']['failed_requests']}
  Success Rate: {analysis['summary']['success_rate_percent']:.1f}%

Performance:
  Avg Response Time: {analysis['performance']['avg_response_time_ms']:.1f}ms
  P95 Response Time: {analysis['performance']['p95_response_time_ms']:.1f}ms
  P99 Response Time: {analysis['performance']['p99_response_time_ms']:.1f}ms

Load Balancing:
  Balance Score: {analysis['load_balancing']['load_balance_effectiveness']:.1f}%
  Backend Distribution:""")
    
    for backend_id, stats in analysis['load_balancing']['backend_distribution'].items():
        print(f"    {backend_id}: {stats['requests']} requests, {stats['avg_time']:.1f}ms avg")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🚀 Instrucciones de Ejecución

### 1. **Preparar el Entorno**

```bash
# Crear directorio del proyecto
mkdir llm-load-balancer
cd llm-load-balancer

# Crear estructura de directorios
mkdir -p load_balancer llm_instance auto_scaler grafana/dashboards grafana/datasources

# Copiar archivos de código
# (copiar todos los archivos Python a sus respectivos directorios)
```

### 2. **Configurar Requirements**

```txt
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
asyncpg==0.29.0
aiohttp==3.9.1
openai==1.3.5
prometheus-client==0.19.0
docker==6.1.3
pydantic==2.5.0
```

### 3. **Ejecutar el Sistema**

```bash
# Construir imágenes
docker-compose build

# Iniciar servicios
docker-compose up -d

# Verificar estado
docker-compose ps
```

### 4. **Testing del Sistema**

```bash
# Ejecutar load test
python test_load_balancer.py

# Verificar métricas
curl http://localhost:8000/status

# Ver logs de auto-scaling
docker logs llm_auto_scaler -f
```

### 5. **Monitoreo**

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Load Balancer Status**: http://localhost:8000/status
- **Rate Limit Stats**: http://localhost:8000/rate-limits/user:test_user

## 📊 Dashboards y Alertas

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "id": null,
    "title": "LLM Load Balancer Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m]) * 60",
            "legendFormat": "Requests/min"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(llm_request_duration_seconds_sum[5m]) / rate(llm_request_duration_seconds_count[5m])",
            "legendFormat": "Avg Response Time"
          }
        ]
      },
      {
        "title": "Backend Health",
        "type": "table",
        "targets": [
          {
            "expr": "llm_active_connections",
            "legendFormat": "{{backend}}"
          }
        ]
      }
    ]
  }
}
```

## 🎯 Ejercicios Prácticos

### Ejercicio 1: Rate Limiting
1. Configurar diferentes límites para usuarios premium vs básicos
2. Implementar burst limits para picos de tráfico
3. Crear alertas cuando se alcancen los límites

### Ejercicio 2: Load Balancing
1. Probar diferentes estrategias de balanceo
2. Simular fallos de backends y observar recuperación
3. Optimizar pesos basados en capacidad real

### Ejercicio 3: Auto-scaling
1. Configurar thresholds más agresivos
2. Implementar predictive scaling basado en patrones históricos
3. Integrar con alertas de Prometheus

## ✅ Criterios de Éxito

- ✅ Load balancer distribuye correctamente las requests
- ✅ Rate limiting funciona para diferentes tipos de usuarios
- ✅ Auto-scaling responde a cambios de carga
- ✅ Circuit breakers protegen contra fallos
- ✅ Métricas se capturan correctamente en Prometheus
- ✅ Dashboards muestran información relevante

## 🔧 Troubleshooting

### Problemas Comunes

1. **Containers no inician**: Verificar logs con `docker logs <container>`
2. **Rate limiting muy estricto**: Ajustar valores en `AdvancedRateLimiter`
3. **Auto-scaling no funciona**: Verificar conectividad con Prometheus
4. **Load balancer no encuentra backends**: Verificar health checks

### Comandos Útiles

```bash
# Ver estado de todos los servicios
docker-compose ps

# Logs en tiempo real
docker-compose logs -f load_balancer

# Reiniciar un servicio específico
docker-compose restart llm_instance_1

# Escalar manualmente
docker-compose up --scale llm_instance=3

# Limpiar todo
docker-compose down -v
```

¡Excelente trabajo! Has implementado un sistema completo de load balancing, rate limiting y auto-scaling para aplicaciones LLM. 🚀
