# ⚡ Lección 2: Performance y Escalabilidad

## 🎯 Objetivos de la Lección

Al finalizar esta lección, serás capaz de:
- Diseñar arquitecturas escalables para aplicaciones LLM
- Implementar load balancing y rate limiting
- Optimizar throughput y latencia
- Configurar auto-scaling basado en métricas
- Gestionar pools de conexiones y circuit breakers
- Implementar streaming y procesamiento asíncrono

## 📊 Fundamentos de Performance en LLMs

### 1. Métricas Clave de Performance

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import time
import statistics
from enum import Enum

class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_DEPTH = "queue_depth"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    TOKEN_RATE = "token_rate"

@dataclass
class PerformanceMetric:
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: str

class PerformanceMonitor:
    """Monitor de performance para sistemas LLM"""
    
    def __init__(self):
        self.metrics_buffer = []
        self.request_history = []
        
        # Configuración de SLAs
        self.sla_targets = {
            MetricType.LATENCY: 2.0,  # 2 segundos max
            MetricType.THROUGHPUT: 100,  # 100 requests/min min
            MetricType.ERROR_RATE: 0.01,  # 1% max
            MetricType.CPU_USAGE: 0.8,   # 80% max
            MetricType.MEMORY_USAGE: 0.85  # 85% max
        }
    
    async def record_request(self, 
                           request_id: str,
                           model_name: str,
                           input_tokens: int,
                           output_tokens: int,
                           duration_ms: float,
                           success: bool) -> Dict[str, Any]:
        """Registrar métricas de una request"""
        
        timestamp = datetime.now()
        
        request_record = {
            'request_id': request_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'duration_ms': duration_ms,
            'success': success,
            'tokens_per_second': (input_tokens + output_tokens) / (duration_ms / 1000) if duration_ms > 0 else 0
        }
        
        self.request_history.append(request_record)
        
        # Mantener solo últimas 1000 requests en memoria
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
        
        # Registrar métricas específicas
        await self._record_metric(MetricType.LATENCY, duration_ms / 1000, {'model': model_name}, 'seconds')
        await self._record_metric(MetricType.TOKEN_RATE, request_record['tokens_per_second'], {'model': model_name}, 'tokens/sec')
        
        return request_record
    
    async def _record_metric(self, 
                           metric_type: MetricType, 
                           value: float, 
                           labels: Dict[str, str],
                           unit: str):
        """Registrar una métrica específica"""
        
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            unit=unit
        )
        
        self.metrics_buffer.append(metric)
        
        # Mantener buffer limitado
        if len(self.metrics_buffer) > 10000:
            self.metrics_buffer = self.metrics_buffer[-5000:]
    
    def get_performance_summary(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Obtener resumen de performance"""
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_requests = [
            r for r in self.request_history 
            if r['timestamp'] >= cutoff_time
        ]
        
        if not recent_requests:
            return {'message': 'No recent requests found'}
        
        # Calcular métricas agregadas
        latencies = [r['duration_ms'] / 1000 for r in recent_requests]
        successful_requests = [r for r in recent_requests if r['success']]
        error_count = len(recent_requests) - len(successful_requests)
        
        throughput = len(recent_requests) / time_window_minutes  # requests per minute
        error_rate = error_count / len(recent_requests) if recent_requests else 0
        
        token_rates = [r['tokens_per_second'] for r in recent_requests if r['tokens_per_second'] > 0]
        
        summary = {
            'time_window_minutes': time_window_minutes,
            'total_requests': len(recent_requests),
            'successful_requests': len(successful_requests),
            'error_count': error_count,
            'latency_metrics': {
                'avg_seconds': round(statistics.mean(latencies), 3),
                'p50_seconds': round(statistics.median(latencies), 3),
                'p95_seconds': round(self._percentile(latencies, 95), 3),
                'p99_seconds': round(self._percentile(latencies, 99), 3),
                'max_seconds': round(max(latencies), 3)
            },
            'throughput_metrics': {
                'requests_per_minute': round(throughput, 2),
                'requests_per_second': round(throughput / 60, 2)
            },
            'error_metrics': {
                'error_rate_percent': round(error_rate * 100, 2),
                'errors_per_minute': round(error_count / time_window_minutes, 2)
            },
            'token_metrics': {
                'avg_tokens_per_second': round(statistics.mean(token_rates), 2) if token_rates else 0,
                'max_tokens_per_second': round(max(token_rates), 2) if token_rates else 0
            }
        }
        
        # Verificar SLAs
        summary['sla_compliance'] = self._check_sla_compliance(summary)
        
        return summary
    
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
    
    def _check_sla_compliance(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Verificar cumplimiento de SLAs"""
        
        compliance = {}
        
        # Verificar latencia
        avg_latency = summary['latency_metrics']['avg_seconds']
        latency_target = self.sla_targets[MetricType.LATENCY]
        compliance['latency'] = {
            'compliant': avg_latency <= latency_target,
            'current': avg_latency,
            'target': latency_target,
            'variance_percent': round(((avg_latency - latency_target) / latency_target) * 100, 2)
        }
        
        # Verificar throughput
        current_throughput = summary['throughput_metrics']['requests_per_minute']
        throughput_target = self.sla_targets[MetricType.THROUGHPUT]
        compliance['throughput'] = {
            'compliant': current_throughput >= throughput_target,
            'current': current_throughput,
            'target': throughput_target,
            'variance_percent': round(((current_throughput - throughput_target) / throughput_target) * 100, 2)
        }
        
        # Verificar error rate
        current_error_rate = summary['error_metrics']['error_rate_percent'] / 100
        error_rate_target = self.sla_targets[MetricType.ERROR_RATE]
        compliance['error_rate'] = {
            'compliant': current_error_rate <= error_rate_target,
            'current': current_error_rate,
            'target': error_rate_target,
            'variance_percent': round(((current_error_rate - error_rate_target) / error_rate_target) * 100, 2) if error_rate_target > 0 else 0
        }
        
        # SLA general
        all_compliant = all(metric['compliant'] for metric in compliance.values())
        compliance['overall'] = {
            'compliant': all_compliant,
            'compliance_score': sum(1 for metric in compliance.values() if metric['compliant']) / len(compliance)
        }
        
        return compliance
```

### 2. Load Balancer Inteligente

```python
import random
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import heapq

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE = "adaptive"

@dataclass
class BackendServer:
    id: str
    url: str
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    healthy: bool = True
    circuit_breaker_open: bool = False

class IntelligentLoadBalancer:
    """Load balancer inteligente para APIs de LLM"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.backends: List[BackendServer] = []
        self.round_robin_index = 0
        
        # Configuración de circuit breaker
        self.circuit_breaker_config = {
            'failure_threshold': 5,  # Fallos consecutivos para abrir
            'timeout_seconds': 30,   # Tiempo antes de reintentar
            'success_threshold': 3   # Éxitos para cerrar
        }
        
        # Métricas
        self.performance_monitor = PerformanceMonitor()
        
        # Health check
        self.health_check_interval = 30  # segundos
        self.health_check_task = None
    
    def add_backend(self, backend: BackendServer):
        """Añadir servidor backend"""
        self.backends.append(backend)
        print(f"Added backend: {backend.id} at {backend.url}")
    
    def remove_backend(self, backend_id: str):
        """Remover servidor backend"""
        self.backends = [b for b in self.backends if b.id != backend_id]
        print(f"Removed backend: {backend_id}")
    
    async def select_backend(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[BackendServer]:
        """Seleccionar backend óptimo basado en estrategia"""
        
        # Filtrar backends saludables
        healthy_backends = [
            b for b in self.backends 
            if b.healthy and not b.circuit_breaker_open
        ]
        
        if not healthy_backends:
            # Si no hay backends saludables, intentar con circuit breakers cerrados
            recovery_backends = [
                b for b in self.backends 
                if b.healthy
            ]
            
            if recovery_backends:
                # Intentar recuperar circuit breakers
                for backend in recovery_backends:
                    if backend.circuit_breaker_open:
                        await self._try_close_circuit_breaker(backend)
                
                healthy_backends = [
                    b for b in recovery_backends 
                    if not b.circuit_breaker_open
                ]
        
        if not healthy_backends:
            return None
        
        # Aplicar estrategia de selección
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_backends)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_backends)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_backends)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_backends)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_select(healthy_backends, request_context)
        else:
            return random.choice(healthy_backends)
    
    def _round_robin_select(self, backends: List[BackendServer]) -> BackendServer:
        """Selección round-robin simple"""
        backend = backends[self.round_robin_index % len(backends)]
        self.round_robin_index += 1
        return backend
    
    def _weighted_round_robin_select(self, backends: List[BackendServer]) -> BackendServer:
        """Selección round-robin con pesos"""
        # Crear lista expandida basada en pesos
        weighted_backends = []
        for backend in backends:
            count = max(1, int(backend.weight * 10))
            weighted_backends.extend([backend] * count)
        
        backend = weighted_backends[self.round_robin_index % len(weighted_backends)]
        self.round_robin_index += 1
        return backend
    
    def _least_connections_select(self, backends: List[BackendServer]) -> BackendServer:
        """Seleccionar backend con menos conexiones activas"""
        return min(backends, key=lambda b: b.current_connections)
    
    def _least_response_time_select(self, backends: List[BackendServer]) -> BackendServer:
        """Seleccionar backend con menor tiempo de respuesta promedio"""
        return min(backends, key=lambda b: b.avg_response_time)
    
    def _adaptive_select(self, backends: List[BackendServer], context: Optional[Dict[str, Any]] = None) -> BackendServer:
        """Selección adaptiva basada en múltiples factores"""
        
        scores = []
        
        for backend in backends:
            # Factor de conexiones (menor es mejor)
            connection_factor = 1.0 - (backend.current_connections / max(backend.max_connections, 1))
            
            # Factor de tiempo de respuesta (menor es mejor)
            max_response_time = max(b.avg_response_time for b in backends) or 1.0
            response_time_factor = 1.0 - (backend.avg_response_time / max_response_time)
            
            # Factor de error rate (menor es mejor)
            error_rate = backend.error_count / max(backend.total_requests, 1)
            error_factor = 1.0 - min(error_rate, 1.0)
            
            # Factor de peso
            weight_factor = backend.weight
            
            # Score compuesto
            composite_score = (
                connection_factor * 0.3 +
                response_time_factor * 0.3 +
                error_factor * 0.2 +
                weight_factor * 0.2
            )
            
            scores.append((composite_score, backend))
        
        # Seleccionar el mejor score
        scores.sort(reverse=True)
        return scores[0][1]
    
    async def execute_request(self, 
                            request_func: Callable,
                            *args, 
                            max_retries: int = 3,
                            **kwargs) -> Dict[str, Any]:
        """Ejecutar request con load balancing y retry logic"""
        
        for attempt in range(max_retries + 1):
            backend = await self.select_backend(kwargs.get('context'))
            
            if not backend:
                return {
                    'success': False,
                    'error': 'No healthy backends available',
                    'attempt': attempt + 1
                }
            
            start_time = time.time()
            request_id = f"{backend.id}_{int(start_time * 1000)}"
            
            try:
                # Incrementar conexiones activas
                backend.current_connections += 1
                
                # Ejecutar request
                result = await request_func(backend.url, *args, **kwargs)
                
                # Medir tiempo de respuesta
                response_time = (time.time() - start_time) * 1000
                
                # Actualizar métricas del backend
                await self._update_backend_metrics(backend, response_time, True)
                
                # Registrar en monitor de performance
                await self.performance_monitor.record_request(
                    request_id=request_id,
                    model_name=kwargs.get('model_name', 'unknown'),
                    input_tokens=kwargs.get('input_tokens', 0),
                    output_tokens=kwargs.get('output_tokens', 0),
                    duration_ms=response_time,
                    success=True
                )
                
                return {
                    'success': True,
                    'result': result,
                    'backend_id': backend.id,
                    'response_time_ms': response_time,
                    'attempt': attempt + 1
                }
                
            except Exception as e:
                # Medir tiempo hasta el error
                error_time = (time.time() - start_time) * 1000
                
                # Actualizar métricas del backend
                await self._update_backend_metrics(backend, error_time, False)
                
                # Verificar si abrir circuit breaker
                await self._check_circuit_breaker(backend)
                
                # Registrar error en monitor
                await self.performance_monitor.record_request(
                    request_id=request_id,
                    model_name=kwargs.get('model_name', 'unknown'),
                    input_tokens=kwargs.get('input_tokens', 0),
                    output_tokens=kwargs.get('output_tokens', 0),
                    duration_ms=error_time,
                    success=False
                )
                
                print(f"Request failed on backend {backend.id}: {e}")
                
                # Si es el último intento, devolver error
                if attempt == max_retries:
                    return {
                        'success': False,
                        'error': str(e),
                        'backend_id': backend.id,
                        'total_attempts': attempt + 1
                    }
                
                # Esperar antes del retry
                await asyncio.sleep(min(2 ** attempt, 10))
                
            finally:
                # Decrementar conexiones activas
                backend.current_connections = max(0, backend.current_connections - 1)
    
    async def _update_backend_metrics(self, backend: BackendServer, response_time: float, success: bool):
        """Actualizar métricas del backend"""
        
        backend.total_requests += 1
        
        if success:
            # Actualizar tiempo de respuesta promedio (moving average)
            alpha = 0.1  # Factor de suavizado
            backend.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * backend.avg_response_time
            )
        else:
            backend.error_count += 1
    
    async def _check_circuit_breaker(self, backend: BackendServer):
        """Verificar si abrir circuit breaker"""
        
        if backend.circuit_breaker_open:
            return
        
        # Calcular error rate reciente
        recent_requests = max(backend.total_requests, 10)  # Mínimo 10 requests
        error_rate = backend.error_count / recent_requests
        
        # Abrir circuit breaker si se excede el threshold
        if error_rate >= 0.5:  # 50% error rate
            backend.circuit_breaker_open = True
            backend.circuit_breaker_opened_at = datetime.now()
            print(f"Circuit breaker opened for backend {backend.id}")
    
    async def _try_close_circuit_breaker(self, backend: BackendServer):
        """Intentar cerrar circuit breaker"""
        
        if not backend.circuit_breaker_open:
            return
        
        opened_at = getattr(backend, 'circuit_breaker_opened_at', datetime.now())
        timeout = timedelta(seconds=self.circuit_breaker_config['timeout_seconds'])
        
        if datetime.now() - opened_at > timeout:
            # Realizar health check
            if await self._health_check_backend(backend):
                backend.circuit_breaker_open = False
                backend.error_count = 0  # Reset contador de errores
                print(f"Circuit breaker closed for backend {backend.id}")
    
    async def _health_check_backend(self, backend: BackendServer) -> bool:
        """Realizar health check en un backend"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{backend.url}/health", 
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    healthy = response.status == 200
                    backend.healthy = healthy
                    backend.last_health_check = datetime.now()
                    return healthy
                    
        except Exception as e:
            print(f"Health check failed for backend {backend.id}: {e}")
            backend.healthy = False
            backend.last_health_check = datetime.now()
            return False
    
    async def start_health_monitoring(self):
        """Iniciar monitoreo de salud periódico"""
        
        async def health_check_loop():
            while True:
                for backend in self.backends:
                    await self._health_check_backend(backend)
                    
                    # Intentar cerrar circuit breakers
                    if backend.circuit_breaker_open:
                        await self._try_close_circuit_breaker(backend)
                
                await asyncio.sleep(self.health_check_interval)
        
        self.health_check_task = asyncio.create_task(health_check_loop())
    
    def stop_health_monitoring(self):
        """Detener monitoreo de salud"""
        if self.health_check_task:
            self.health_check_task.cancel()
    
    def get_backend_status(self) -> Dict[str, Any]:
        """Obtener estado de todos los backends"""
        
        backend_stats = []
        
        for backend in self.backends:
            error_rate = backend.error_count / max(backend.total_requests, 1)
            
            stats = {
                'id': backend.id,
                'url': backend.url,
                'healthy': backend.healthy,
                'circuit_breaker_open': backend.circuit_breaker_open,
                'current_connections': backend.current_connections,
                'max_connections': backend.max_connections,
                'total_requests': backend.total_requests,
                'error_count': backend.error_count,
                'error_rate_percent': round(error_rate * 100, 2),
                'avg_response_time_ms': round(backend.avg_response_time, 2),
                'weight': backend.weight,
                'last_health_check': backend.last_health_check.isoformat() if backend.last_health_check else None
            }
            
            backend_stats.append(stats)
        
        healthy_count = sum(1 for b in self.backends if b.healthy and not b.circuit_breaker_open)
        
        return {
            'strategy': self.strategy.value,
            'total_backends': len(self.backends),
            'healthy_backends': healthy_count,
            'backends': backend_stats,
            'overall_health': healthy_count / max(len(self.backends), 1)
        }
```

### 3. Rate Limiting Avanzado

```python
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis

class RateLimitStrategy(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"

@dataclass
class RateLimitRule:
    identifier: str  # user_id, api_key, ip, etc.
    max_requests: int
    time_window_seconds: int
    burst_limit: Optional[int] = None
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

class AdvancedRateLimiter:
    """Rate limiter avanzado con múltiples estrategias"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Configuración por defecto
        self.default_rules = {
            'global': RateLimitRule('global', 1000, 60),  # 1000 req/min global
            'per_user': RateLimitRule('user', 100, 60),   # 100 req/min por usuario
            'per_ip': RateLimitRule('ip', 200, 60),       # 200 req/min por IP
            'premium_user': RateLimitRule('premium', 500, 60)  # 500 req/min premium
        }
        
        # Configuración adaptiva
        self.adaptive_config = {
            'base_limit': 100,
            'max_limit': 1000,
            'increase_factor': 1.2,
            'decrease_factor': 0.8,
            'success_threshold': 0.95,
            'adjustment_interval': 300  # 5 minutos
        }
    
    async def check_rate_limit(self, 
                             identifier: str, 
                             rule_type: str = 'per_user',
                             custom_rule: Optional[RateLimitRule] = None) -> Dict[str, Any]:
        """Verificar si la request está dentro del rate limit"""
        
        rule = custom_rule or self.default_rules.get(rule_type)
        if not rule:
            return {'allowed': True, 'message': 'No rate limit rule defined'}
        
        if rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(identifier, rule)
        elif rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(identifier, rule)
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(identifier, rule)
        elif rule.strategy == RateLimitStrategy.ADAPTIVE:
            return await self._check_adaptive_limit(identifier, rule)
        else:
            return {'allowed': True, 'message': 'Unknown rate limit strategy'}
    
    async def _check_token_bucket(self, identifier: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Implementar algoritmo de token bucket"""
        
        key = f"rate_limit:token_bucket:{identifier}"
        now = time.time()
        
        # Obtener estado actual del bucket
        bucket_data = await self.redis.hgetall(key)
        
        if bucket_data:
            last_refill = float(bucket_data.get(b'last_refill', now))
            tokens = float(bucket_data.get(b'tokens', rule.max_requests))
        else:
            last_refill = now
            tokens = rule.max_requests
        
        # Calcular tokens a añadir
        time_passed = now - last_refill
        refill_rate = rule.max_requests / rule.time_window_seconds
        tokens_to_add = time_passed * refill_rate
        tokens = min(rule.max_requests, tokens + tokens_to_add)
        
        # Verificar si hay tokens disponibles
        if tokens >= 1:
            tokens -= 1
            allowed = True
            
            # Actualizar estado del bucket
            await self.redis.hset(key, mapping={
                'tokens': str(tokens),
                'last_refill': str(now)
            })
            await self.redis.expire(key, rule.time_window_seconds * 2)
            
        else:
            allowed = False
        
        return {
            'allowed': allowed,
            'strategy': 'token_bucket',
            'tokens_remaining': int(tokens),
            'refill_rate': refill_rate,
            'retry_after_seconds': (1 - tokens) / refill_rate if not allowed else 0
        }
    
    async def _check_sliding_window(self, identifier: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Implementar algoritmo de sliding window"""
        
        key = f"rate_limit:sliding:{identifier}"
        now = time.time()
        window_start = now - rule.time_window_seconds
        
        # Usar Redis sorted set para sliding window
        pipe = self.redis.pipeline()
        
        # Remover requests antiguos
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Contar requests en la ventana
        pipe.zcard(key)
        
        # Añadir request actual
        pipe.zadd(key, {str(now): now})
        
        # Configurar expiración
        pipe.expire(key, rule.time_window_seconds)
        
        results = await pipe.execute()
        current_count = results[1]
        
        allowed = current_count < rule.max_requests
        
        if not allowed:
            # Remover la request que acabamos de añadir
            await self.redis.zrem(key, str(now))
        
        # Calcular tiempo hasta el próximo slot disponible
        retry_after = 0
        if not allowed:
            # Obtener la request más antigua en la ventana
            oldest_requests = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest_requests:
                oldest_time = oldest_requests[0][1]
                retry_after = oldest_time + rule.time_window_seconds - now
        
        return {
            'allowed': allowed,
            'strategy': 'sliding_window',
            'current_count': current_count,
            'limit': rule.max_requests,
            'window_seconds': rule.time_window_seconds,
            'retry_after_seconds': max(0, retry_after)
        }
    
    async def _check_fixed_window(self, identifier: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Implementar algoritmo de fixed window"""
        
        now = time.time()
        window = int(now // rule.time_window_seconds)
        key = f"rate_limit:fixed:{identifier}:{window}"
        
        # Incrementar contador
        current_count = await self.redis.incr(key)
        
        # Configurar expiración solo en la primera request de la ventana
        if current_count == 1:
            await self.redis.expire(key, rule.time_window_seconds)
        
        allowed = current_count <= rule.max_requests
        
        if not allowed:
            # Decrementar si no está permitido
            await self.redis.decr(key)
            current_count -= 1
        
        # Calcular tiempo hasta la próxima ventana
        next_window_start = (window + 1) * rule.time_window_seconds
        retry_after = next_window_start - now
        
        return {
            'allowed': allowed,
            'strategy': 'fixed_window',
            'current_count': current_count,
            'limit': rule.max_requests,
            'window_seconds': rule.time_window_seconds,
            'retry_after_seconds': retry_after if not allowed else 0
        }
    
    async def _check_adaptive_limit(self, identifier: str, rule: RateLimitRule) -> Dict[str, Any]:
        """Implementar rate limiting adaptivo"""
        
        # Obtener límite adaptivo actual
        adaptive_key = f"rate_limit:adaptive:{identifier}"
        current_limit_data = await self.redis.hgetall(adaptive_key)
        
        if current_limit_data:
            current_limit = int(current_limit_data.get(b'limit', self.adaptive_config['base_limit']))
            last_adjustment = float(current_limit_data.get(b'last_adjustment', time.time()))
            success_rate = float(current_limit_data.get(b'success_rate', 1.0))
        else:
            current_limit = self.adaptive_config['base_limit']
            last_adjustment = time.time()
            success_rate = 1.0
        
        # Crear regla temporal con límite adaptivo
        adaptive_rule = RateLimitRule(
            identifier=identifier,
            max_requests=current_limit,
            time_window_seconds=rule.time_window_seconds,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        
        # Verificar con límite adaptivo
        result = await self._check_sliding_window(identifier, adaptive_rule)
        
        # Ajustar límite si es tiempo
        now = time.time()
        if now - last_adjustment >= self.adaptive_config['adjustment_interval']:
            new_limit = await self._adjust_adaptive_limit(
                identifier, current_limit, success_rate
            )
            
            await self.redis.hset(adaptive_key, mapping={
                'limit': str(new_limit),
                'last_adjustment': str(now),
                'success_rate': str(success_rate)
            })
            await self.redis.expire(adaptive_key, rule.time_window_seconds * 10)
        
        result['adaptive_limit'] = current_limit
        result['strategy'] = 'adaptive'
        
        return result
    
    async def _adjust_adaptive_limit(self, identifier: str, current_limit: int, success_rate: float) -> int:
        """Ajustar límite adaptivo basado en métricas"""
        
        base_limit = self.adaptive_config['base_limit']
        max_limit = self.adaptive_config['max_limit']
        success_threshold = self.adaptive_config['success_threshold']
        
        if success_rate >= success_threshold:
            # Incrementar límite si la success rate es alta
            new_limit = min(
                max_limit,
                int(current_limit * self.adaptive_config['increase_factor'])
            )
        else:
            # Decrementar límite si hay muchos errores
            new_limit = max(
                base_limit,
                int(current_limit * self.adaptive_config['decrease_factor'])
            )
        
        return new_limit
    
    async def update_success_rate(self, identifier: str, success: bool):
        """Actualizar success rate para rate limiting adaptivo"""
        
        key = f"rate_limit:adaptive:{identifier}"
        
        # Usar moving average para success rate
        current_data = await self.redis.hgetall(key)
        
        if current_data:
            current_rate = float(current_data.get(b'success_rate', 1.0))
        else:
            current_rate = 1.0
        
        # Factor de suavizado para moving average
        alpha = 0.1
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
        
        await self.redis.hset(key, 'success_rate', str(new_rate))
    
    async def get_rate_limit_stats(self, identifier: str) -> Dict[str, Any]:
        """Obtener estadísticas de rate limiting"""
        
        stats = {}
        
        # Verificar todos los tipos de rate limit
        for rule_type, rule in self.default_rules.items():
            result = await self.check_rate_limit(identifier, rule_type)
            stats[rule_type] = result
        
        # Añadir estadísticas adicionales
        stats['identifier'] = identifier
        stats['timestamp'] = datetime.now().isoformat()
        
        return stats
    
    async def reset_rate_limit(self, identifier: str, rule_type: Optional[str] = None):
        """Resetear rate limit para un identificador"""
        
        if rule_type:
            # Resetear tipo específico
            patterns = [
                f"rate_limit:*:{identifier}*",
                f"rate_limit:adaptive:{identifier}"
            ]
        else:
            # Resetear todos los tipos
            patterns = [f"rate_limit:*:{identifier}*"]
        
        for pattern in patterns:
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
    
    def create_custom_rule(self, 
                          identifier: str,
                          max_requests: int,
                          time_window_seconds: int,
                          strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
                          burst_limit: Optional[int] = None) -> RateLimitRule:
        """Crear regla personalizada de rate limiting"""
        
        return RateLimitRule(
            identifier=identifier,
            max_requests=max_requests,
            time_window_seconds=time_window_seconds,
            burst_limit=burst_limit,
            strategy=strategy
        )
```

## 🚀 Auto-scaling y Gestión de Recursos

### 1. Auto-scaler Inteligente

```python
import asyncio
import math
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

@dataclass
class ScalingMetric:
    name: str
    current_value: float
    target_value: float
    weight: float
    threshold_up: float
    threshold_down: float

class IntelligentAutoScaler:
    """Auto-scaler inteligente para aplicaciones LLM"""
    
    def __init__(self):
        self.current_instances = 1
        self.min_instances = 1
        self.max_instances = 10
        self.scaling_cooldown = 300  # 5 minutos
        self.last_scaling_action = None
        
        # Métricas de scaling
        self.metrics = {
            'cpu_usage': ScalingMetric('cpu_usage', 0.0, 0.7, 0.3, 0.8, 0.5),
            'memory_usage': ScalingMetric('memory_usage', 0.0, 0.8, 0.2, 0.85, 0.6),
            'request_rate': ScalingMetric('request_rate', 0.0, 100.0, 0.3, 150.0, 50.0),
            'response_time': ScalingMetric('response_time', 0.0, 2.0, 0.2, 3.0, 1.0)
        }
        
        # Configuración predictiva
        self.predictive_scaling = True
        self.prediction_window = 600  # 10 minutos
        
    async def evaluate_scaling_decision(self, 
                                      current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluar si es necesario escalar"""
        
        # Actualizar métricas actuales
        for metric_name, value in current_metrics.items():
            if metric_name in self.metrics:
                self.metrics[metric_name].current_value = value
        
        # Verificar cooldown
        if self._is_in_cooldown():
            return {
                'action': ScalingDirection.STABLE,
                'reason': 'Scaling cooldown active',
                'current_instances': self.current_instances
            }
        
        # Calcular scores de scaling
        scale_up_score = self._calculate_scale_up_score()
        scale_down_score = self._calculate_scale_down_score()
        
        # Decidir acción
        if scale_up_score > 0.7:
            action = ScalingDirection.UP
            target_instances = self._calculate_target_instances_up()
            reason = f"Scale up triggered (score: {scale_up_score:.2f})"
        elif scale_down_score > 0.7:
            action = ScalingDirection.DOWN
            target_instances = self._calculate_target_instances_down()
            reason = f"Scale down triggered (score: {scale_down_score:.2f})"
        else:
            action = ScalingDirection.STABLE
            target_instances = self.current_instances
            reason = "Metrics within acceptable range"
        
        # Aplicar predicción si está habilitada
        if self.predictive_scaling and action == ScalingDirection.STABLE:
            prediction = await self._predict_future_load()
            if prediction['should_preemptive_scale']:
                action = prediction['direction']
                target_instances = prediction['target_instances']
                reason = f"Predictive scaling: {prediction['reason']}"
        
        return {
            'action': action,
            'current_instances': self.current_instances,
            'target_instances': target_instances,
            'reason': reason,
            'scale_up_score': scale_up_score,
            'scale_down_score': scale_down_score,
            'metrics_analysis': self._analyze_metrics(),
            'estimated_impact': self._estimate_scaling_impact(target_instances)
        }
    
    def _calculate_scale_up_score(self) -> float:
        """Calcular score para escalar hacia arriba"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric in self.metrics.values():
            if metric.current_value > metric.threshold_up:
                # Calcular qué tan por encima del threshold está
                excess = (metric.current_value - metric.threshold_up) / metric.threshold_up
                metric_score = min(1.0, excess)  # Cap en 1.0
                
                total_score += metric_score * metric.weight
                total_weight += metric.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_scale_down_score(self) -> float:
        """Calcular score para escalar hacia abajo"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric in self.metrics.values():
            if metric.current_value < metric.threshold_down:
                # Calcular qué tan por debajo del threshold está
                deficit = (metric.threshold_down - metric.current_value) / metric.threshold_down
                metric_score = min(1.0, deficit)  # Cap en 1.0
                
                total_score += metric_score * metric.weight
                total_weight += metric.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_target_instances_up(self) -> int:
        """Calcular número objetivo de instancias para scale up"""
        
        # Encontrar la métrica más crítica
        max_ratio = 0.0
        for metric in self.metrics.values():
            if metric.target_value > 0:
                ratio = metric.current_value / metric.target_value
                max_ratio = max(max_ratio, ratio)
        
        # Calcular instancias necesarias basándose en la métrica más crítica
        if max_ratio > 1.0:
            needed_instances = math.ceil(self.current_instances * max_ratio)
            return min(needed_instances, self.max_instances)
        else:
            # Scale up conservador
            return min(self.current_instances + 1, self.max_instances)
    
    def _calculate_target_instances_down(self) -> int:
        """Calcular número objetivo de instancias para scale down"""
        
        # Encontrar la métrica con menor utilización
        min_ratio = float('inf')
        for metric in self.metrics.values():
            if metric.target_value > 0 and metric.current_value > 0:
                ratio = metric.current_value / metric.target_value
                min_ratio = min(min_ratio, ratio)
        
        # Calcular instancias necesarias basándose en utilización mínima
        if min_ratio < 1.0 and min_ratio != float('inf'):
            needed_instances = max(1, math.floor(self.current_instances * min_ratio))
            return max(needed_instances, self.min_instances)
        else:
            # Scale down conservador
            return max(self.current_instances - 1, self.min_instances)
    
    def _is_in_cooldown(self) -> bool:
        """Verificar si estamos en periodo de cooldown"""
        
        if not self.last_scaling_action:
            return False
        
        elapsed = (datetime.now() - self.last_scaling_action).total_seconds()
        return elapsed < self.scaling_cooldown
    
    def _analyze_metrics(self) -> Dict[str, Any]:
        """Analizar estado actual de las métricas"""
        
        analysis = {}
        
        for name, metric in self.metrics.items():
            if metric.target_value > 0:
                utilization = metric.current_value / metric.target_value
                status = "normal"
                
                if metric.current_value > metric.threshold_up:
                    status = "high"
                elif metric.current_value < metric.threshold_down:
                    status = "low"
                
                analysis[name] = {
                    'current': metric.current_value,
                    'target': metric.target_value,
                    'utilization_percent': round(utilization * 100, 1),
                    'status': status,
                    'threshold_up': metric.threshold_up,
                    'threshold_down': metric.threshold_down
                }
        
        return analysis
    
    def _estimate_scaling_impact(self, target_instances: int) -> Dict[str, Any]:
        """Estimar impacto del scaling"""
        
        if target_instances == self.current_instances:
            return {'no_change': True}
        
        scaling_factor = target_instances / self.current_instances
        
        # Estimar nuevos valores de métricas
        estimated_metrics = {}
        for name, metric in self.metrics.items():
            if name in ['cpu_usage', 'memory_usage']:
                # CPU y memoria se distribuyen entre instancias
                new_value = metric.current_value / scaling_factor
            elif name == 'request_rate':
                # Request rate se mantiene igual (se distribuye)
                new_value = metric.current_value
            elif name == 'response_time':
                # Response time mejora con más instancias
                new_value = metric.current_value / math.sqrt(scaling_factor)
            else:
                new_value = metric.current_value
            
            estimated_metrics[name] = round(new_value, 3)
        
        return {
            'scaling_factor': round(scaling_factor, 2),
            'estimated_metrics': estimated_metrics,
            'cost_impact_percent': round((scaling_factor - 1) * 100, 1),
            'performance_improvement': scaling_factor > 1
        }
    
    async def _predict_future_load(self) -> Dict[str, Any]:
        """Predicción simple de carga futura"""
        
        # Implementación básica - en producción usar ML más sofisticado
        current_hour = datetime.now().hour
        
        # Patrones simples basados en hora del día
        if 8 <= current_hour <= 18:  # Horario laboral
            load_factor = 1.5
            should_scale = True
            direction = ScalingDirection.UP
            reason = "Peak hours approaching"
        elif 22 <= current_hour or current_hour <= 6:  # Horario nocturno
            load_factor = 0.5
            should_scale = self.current_instances > self.min_instances
            direction = ScalingDirection.DOWN
            reason = "Low activity period"
        else:
            load_factor = 1.0
            should_scale = False
            direction = ScalingDirection.STABLE
            reason = "Normal activity period"
        
        target_instances = max(
            self.min_instances,
            min(self.max_instances, round(self.current_instances * load_factor))
        )
        
        return {
            'should_preemptive_scale': should_scale and target_instances != self.current_instances,
            'direction': direction,
            'target_instances': target_instances,
            'reason': reason,
            'load_factor': load_factor
        }
    
    async def execute_scaling(self, 
                            target_instances: int,
                            scaling_executor: Callable[[int], Any]) -> Dict[str, Any]:
        """Ejecutar acción de scaling"""
        
        if target_instances == self.current_instances:
            return {'executed': False, 'reason': 'No scaling needed'}
        
        try:
            # Ejecutar scaling (función proporcionada por el usuario)
            result = await scaling_executor(target_instances)
            
            # Actualizar estado interno
            old_instances = self.current_instances
            self.current_instances = target_instances
            self.last_scaling_action = datetime.now()
            
            return {
                'executed': True,
                'old_instances': old_instances,
                'new_instances': target_instances,
                'scaling_result': result,
                'timestamp': self.last_scaling_action.isoformat()
            }
            
        except Exception as e:
            return {
                'executed': False,
                'error': str(e),
                'target_instances': target_instances
            }
    
    def get_scaling_history(self) -> Dict[str, Any]:
        """Obtener historial de scaling"""
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'last_scaling_action': self.last_scaling_action.isoformat() if self.last_scaling_action else None,
            'cooldown_remaining': max(0, self.scaling_cooldown - (
                (datetime.now() - self.last_scaling_action).total_seconds() 
                if self.last_scaling_action else self.scaling_cooldown
            )),
            'current_metrics': {name: metric.current_value for name, metric in self.metrics.items()}
        }
```

## ✅ Mejores Prácticas de Performance

### 1. **Optimización de Latencia**
- Usar conexiones persistentes
- Implementar connection pooling
- Cachear responses frecuentes
- Minimizar serialización/deserialización

### 2. **Escalabilidad Horizontal**
- Load balancing inteligente
- Auto-scaling basado en métricas
- Circuit breakers para fallos
- Health checks proactivos

### 3. **Rate Limiting Efectivo**
- Estrategias adaptivas
- Límites por tipo de usuario
- Graceful degradation
- Monitoreo de abuse patterns

### 4. **Monitoreo Continuo**
- SLAs bien definidos
- Alertas proactivas
- Métricas de negocio
- Observabilidad end-to-end

## 🎯 Próximo Paso

En el **Laboratorio 6** implementaremos un sistema completo de load balancing, rate limiting y auto-scaling con métricas en tiempo real.

## 📖 Recursos Adicionales

- [Load Balancing Patterns](https://microservices.io/patterns/deployment/client-side-discovery.html)
- [Rate Limiting Algorithms](https://konghq.com/blog/how-to-design-a-scalable-rate-limiting-algorithm)
- [Auto-scaling Best Practices](https://aws.amazon.com/autoscaling/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
