# 🏥 Lección 2: Health Checks y Service Discovery

## 🎯 Objetivos de la Lección

Al finalizar esta lección, serás capaz de:
- Implementar health checks comprensivos para LLMs
- Diseñar service discovery dinámico
- Configurar dependency health monitoring
- Crear health aggregation inteligente
- Implementar graceful shutdown patterns
- Gestionar rolling deployments sin downtime

## 🩺 Health Checks Avanzados

### 1. Sistema Comprensivo de Health Checks

```python
import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import psutil
import aiohttp

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    details: Optional[Dict[str, Any]] = None

class HealthCheck(ABC):
    """Interfaz base para health checks"""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"health.{name}")
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Ejecutar health check"""
        pass
    
    async def execute_with_timeout(self) -> HealthCheckResult:
        """Ejecutar health check con timeout"""
        
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.check(),
                timeout=self.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout_seconds}s",
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={'timeout': True}
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={'error': str(e), 'exception_type': type(e).__name__}
            )

class LLMServiceHealthCheck(HealthCheck):
    """Health check para servicios LLM"""
    
    def __init__(self, name: str, llm_client, test_prompt: str = "Hello"):
        super().__init__(name)
        self.llm_client = llm_client
        self.test_prompt = test_prompt
    
    async def check(self) -> HealthCheckResult:
        """Verificar salud del servicio LLM"""
        
        start_time = time.time()
        
        try:
            # Test de conectividad básica
            response = await self.llm_client.generate(
                prompt=self.test_prompt,
                model="gpt-3.5-turbo",
                max_tokens=10,
                temperature=0.0
            )
            
            duration = (time.time() - start_time) * 1000
            
            # Verificar respuesta válida
            if not response or not response.get('text'):
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="LLM returned empty response",
                    timestamp=datetime.now(),
                    duration_ms=duration,
                    metadata={'response': response}
                )
            
            # Verificar latencia
            if duration > 10000:  # 10 segundos
                status = HealthStatus.DEGRADED
                message = f"LLM response slow: {duration:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"LLM healthy: {duration:.1f}ms"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={
                    'response_length': len(response.get('text', '')),
                    'tokens_used': response.get('tokens_used', 0)
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"LLM service failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={'error_type': type(e).__name__}
            )

class DatabaseHealthCheck(HealthCheck):
    """Health check para base de datos"""
    
    def __init__(self, name: str, db_pool, test_query: str = "SELECT 1"):
        super().__init__(name)
        self.db_pool = db_pool
        self.test_query = test_query
    
    async def check(self) -> HealthCheckResult:
        """Verificar salud de la base de datos"""
        
        start_time = time.time()
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(self.test_query)
            
            duration = (time.time() - start_time) * 1000
            
            # Verificar pool de conexiones
            pool_stats = {
                'size': self.db_pool.get_size(),
                'min_size': self.db_pool.get_min_size(),
                'max_size': self.db_pool.get_max_size(),
                'free_connections': self.db_pool.get_idle_size()
            }
            
            # Evaluar estado basado en disponibilidad del pool
            if pool_stats['free_connections'] == 0:
                status = HealthStatus.DEGRADED
                message = "Database pool exhausted"
            elif duration > 1000:  # 1 segundo
                status = HealthStatus.DEGRADED
                message = f"Database slow: {duration:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy: {duration:.1f}ms"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata=pool_stats
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={'error_type': type(e).__name__}
            )

class RedisHealthCheck(HealthCheck):
    """Health check para Redis"""
    
    def __init__(self, name: str, redis_client):
        super().__init__(name)
        self.redis_client = redis_client
    
    async def check(self) -> HealthCheckResult:
        """Verificar salud de Redis"""
        
        start_time = time.time()
        
        try:
            # Test de conectividad
            await self.redis_client.ping()
            
            # Test de write/read
            test_key = f"health_check_{int(time.time())}"
            await self.redis_client.set(test_key, "test_value", ex=10)
            value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)
            
            duration = (time.time() - start_time) * 1000
            
            if value != "test_value":
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Redis write/read mismatch",
                    timestamp=datetime.now(),
                    duration_ms=duration
                )
            
            # Obtener info de Redis
            info = await self.redis_client.info()
            memory_usage = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            memory_ratio = memory_usage / max_memory if max_memory > 0 else 0
            
            if memory_ratio > 0.9:
                status = HealthStatus.DEGRADED
                message = f"Redis memory high: {memory_ratio:.1%}"
            elif duration > 100:  # 100ms
                status = HealthStatus.DEGRADED
                message = f"Redis slow: {duration:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis healthy: {duration:.1f}ms"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={
                    'memory_usage_bytes': memory_usage,
                    'memory_usage_ratio': memory_ratio,
                    'connected_clients': info.get('connected_clients', 0)
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={'error_type': type(e).__name__}
            )

class SystemResourceHealthCheck(HealthCheck):
    """Health check para recursos del sistema"""
    
    def __init__(self, 
                 name: str,
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0):
        super().__init__(name)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def check(self) -> HealthCheckResult:
        """Verificar recursos del sistema"""
        
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            duration = (time.time() - start_time) * 1000
            
            # Evaluar estado
            issues = []
            if cpu_percent > self.cpu_threshold:
                issues.append(f"CPU high: {cpu_percent:.1f}%")
            
            if memory_percent > self.memory_threshold:
                issues.append(f"Memory high: {memory_percent:.1f}%")
            
            if disk_percent > self.disk_threshold:
                issues.append(f"Disk high: {disk_percent:.1f}%")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = ", ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={'error_type': type(e).__name__}
            )

class ExternalServiceHealthCheck(HealthCheck):
    """Health check para servicios externos"""
    
    def __init__(self, name: str, url: str, expected_status: int = 200):
        super().__init__(name)
        self.url = url
        self.expected_status = expected_status
    
    async def check(self) -> HealthCheckResult:
        """Verificar servicio externo"""
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
                ) as response:
                    duration = (time.time() - start_time) * 1000
                    
                    if response.status == self.expected_status:
                        status = HealthStatus.HEALTHY
                        message = f"External service healthy: {response.status}"
                    else:
                        status = HealthStatus.DEGRADED
                        message = f"External service returned {response.status}, expected {self.expected_status}"
                    
                    return HealthCheckResult(
                        name=self.name,
                        status=status,
                        message=message,
                        timestamp=datetime.now(),
                        duration_ms=duration,
                        metadata={
                            'status_code': response.status,
                            'headers': dict(response.headers),
                            'url': self.url
                        }
                    )
                    
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"External service failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration,
                metadata={'error_type': type(e).__name__, 'url': self.url}
            )

class CompositeHealthChecker:
    """Agregador de múltiples health checks"""
    
    def __init__(self, name: str = "composite"):
        self.name = name
        self.health_checks: List[HealthCheck] = []
        self.logger = logging.getLogger(f"health.{name}")
        
        # Configuración de agregación
        self.critical_checks = set()  # Health checks críticos
        self.weights = {}  # Pesos para cada check
    
    def add_health_check(self, 
                        health_check: HealthCheck, 
                        is_critical: bool = False,
                        weight: float = 1.0):
        """Añadir health check"""
        
        self.health_checks.append(health_check)
        
        if is_critical:
            self.critical_checks.add(health_check.name)
        
        self.weights[health_check.name] = weight
    
    async def check_all(self, parallel: bool = True) -> Dict[str, Any]:
        """Ejecutar todos los health checks"""
        
        start_time = time.time()
        
        if parallel:
            # Ejecutar en paralelo
            tasks = [hc.execute_with_timeout() for hc in self.health_checks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Ejecutar secuencialmente
            results = []
            for hc in self.health_checks:
                result = await hc.execute_with_timeout()
                results.append(result)
        
        duration = (time.time() - start_time) * 1000
        
        # Procesar resultados
        individual_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                hc_name = self.health_checks[i].name
                individual_results[hc_name] = HealthCheckResult(
                    name=hc_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check exception: {str(result)}",
                    timestamp=datetime.now(),
                    duration_ms=0,
                    metadata={'exception': True}
                )
            else:
                individual_results[result.name] = result
        
        # Calcular estado agregado
        overall_status, overall_message = self._calculate_overall_status(individual_results)
        
        return {
            'overall_status': overall_status.value,
            'overall_message': overall_message,
            'timestamp': datetime.now().isoformat(),
            'check_duration_ms': duration,
            'individual_results': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'duration_ms': result.duration_ms,
                    'metadata': result.metadata
                }
                for name, result in individual_results.items()
            },
            'summary': self._generate_summary(individual_results)
        }
    
    def _calculate_overall_status(self, 
                                 results: Dict[str, HealthCheckResult]) -> tuple[HealthStatus, str]:
        """Calcular estado general"""
        
        if not results:
            return HealthStatus.UNKNOWN, "No health checks configured"
        
        # Verificar checks críticos
        critical_failed = []
        for check_name in self.critical_checks:
            if check_name in results:
                result = results[check_name]
                if result.status == HealthStatus.UNHEALTHY:
                    critical_failed.append(check_name)
        
        if critical_failed:
            return (
                HealthStatus.UNHEALTHY,
                f"Critical checks failed: {', '.join(critical_failed)}"
            )
        
        # Contar estados
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        total_weight = 0
        weighted_score = 0
        
        for name, result in results.items():
            status_counts[result.status] += 1
            
            weight = self.weights.get(name, 1.0)
            total_weight += weight
            
            # Asignar puntuación por estado
            if result.status == HealthStatus.HEALTHY:
                weighted_score += weight * 1.0
            elif result.status == HealthStatus.DEGRADED:
                weighted_score += weight * 0.5
            # UNHEALTHY y UNKNOWN = 0 puntos
        
        # Calcular estado general basado en puntuación ponderada
        if total_weight > 0:
            health_ratio = weighted_score / total_weight
            
            if health_ratio >= 0.8:
                overall_status = HealthStatus.HEALTHY
                message = "All systems operational"
            elif health_ratio >= 0.5:
                overall_status = HealthStatus.DEGRADED
                message = f"Some systems degraded ({status_counts[HealthStatus.DEGRADED]} degraded, {status_counts[HealthStatus.UNHEALTHY]} unhealthy)"
            else:
                overall_status = HealthStatus.UNHEALTHY
                message = f"Multiple systems failing ({status_counts[HealthStatus.UNHEALTHY]} unhealthy, {status_counts[HealthStatus.DEGRADED]} degraded)"
        else:
            overall_status = HealthStatus.UNKNOWN
            message = "Unable to determine health status"
        
        return overall_status, message
    
    def _generate_summary(self, results: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Generar resumen de salud"""
        
        total_checks = len(results)
        if total_checks == 0:
            return {'total_checks': 0}
        
        status_counts = {status.value: 0 for status in HealthStatus}
        total_duration = 0
        
        for result in results.values():
            status_counts[result.status.value] += 1
            total_duration += result.duration_ms
        
        return {
            'total_checks': total_checks,
            'healthy_count': status_counts[HealthStatus.HEALTHY.value],
            'degraded_count': status_counts[HealthStatus.DEGRADED.value],
            'unhealthy_count': status_counts[HealthStatus.UNHEALTHY.value],
            'unknown_count': status_counts[HealthStatus.UNKNOWN.value],
            'avg_duration_ms': total_duration / total_checks,
            'health_percentage': (status_counts[HealthStatus.HEALTHY.value] / total_checks) * 100
        }
```

### 2. Service Discovery Dinámico

```python
import asyncio
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

class ServiceStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DRAINING = "draining"
    STARTING = "starting"

@dataclass
class ServiceInstance:
    service_name: str
    instance_id: str
    host: str
    port: int
    health_check_url: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.STARTING
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    registration_time: datetime = field(default_factory=datetime.now)
    version: str = "unknown"
    tags: Set[str] = field(default_factory=set)
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def service_key(self) -> str:
        return f"{self.service_name}:{self.instance_id}"

class ServiceRegistry:
    """Registry centralizado de servicios"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_registry: Dict[str, ServiceInstance] = {}
        self.service_watchers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger("service.registry")
        
        # Configuración
        self.health_check_interval = 30  # segundos
        self.failure_threshold = 3
        self.cleanup_interval = 300  # 5 minutos
        
        # Tareas de background
        self._health_check_task = None
        self._cleanup_task = None
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """Registrar un servicio"""
        
        try:
            service_key = service.service_key
            
            # Registrar localmente
            self.local_registry[service_key] = service
            
            # Registrar en Redis si está disponible
            if self.redis_client:
                await self._register_in_redis(service)
            
            self.logger.info(f"Registered service: {service_key}")
            
            # Notificar watchers
            await self._notify_watchers(service.service_name, 'registered', service)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register service {service.service_key}: {e}")
            return False
    
    async def deregister_service(self, service_name: str, instance_id: str) -> bool:
        """Desregistrar un servicio"""
        
        try:
            service_key = f"{service_name}:{instance_id}"
            
            # Remover localmente
            service = self.local_registry.pop(service_key, None)
            
            # Remover de Redis
            if self.redis_client:
                await self._deregister_from_redis(service_name, instance_id)
            
            self.logger.info(f"Deregistered service: {service_key}")
            
            # Notificar watchers
            if service:
                await self._notify_watchers(service_name, 'deregistered', service)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_key}: {e}")
            return False
    
    async def discover_services(self, 
                              service_name: str,
                              tags: Optional[Set[str]] = None,
                              status_filter: Optional[ServiceStatus] = None) -> List[ServiceInstance]:
        """Descubrir servicios disponibles"""
        
        try:
            # Obtener de Redis si está disponible
            if self.redis_client:
                services = await self._discover_from_redis(service_name)
            else:
                services = [
                    service for service in self.local_registry.values()
                    if service.service_name == service_name
                ]
            
            # Aplicar filtros
            filtered_services = []
            
            for service in services:
                # Filtro por estado
                if status_filter and service.status != status_filter:
                    continue
                
                # Filtro por tags
                if tags and not tags.issubset(service.tags):
                    continue
                
                filtered_services.append(service)
            
            return filtered_services
            
        except Exception as e:
            self.logger.error(f"Failed to discover services for {service_name}: {e}")
            return []
    
    async def get_healthy_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Obtener una instancia saludable del servicio"""
        
        services = await self.discover_services(
            service_name, 
            status_filter=ServiceStatus.AVAILABLE
        )
        
        if not services:
            return None
        
        # Seleccionar instancia con menos fallos consecutivos
        return min(services, key=lambda s: s.consecutive_failures)
    
    async def update_service_status(self, 
                                  service_name: str,
                                  instance_id: str,
                                  status: ServiceStatus,
                                  metadata: Optional[Dict[str, Any]] = None):
        """Actualizar estado de un servicio"""
        
        service_key = f"{service_name}:{instance_id}"
        
        if service_key in self.local_registry:
            service = self.local_registry[service_key]
            service.status = status
            
            if metadata:
                service.metadata.update(metadata)
            
            # Actualizar en Redis
            if self.redis_client:
                await self._update_in_redis(service)
            
            # Notificar watchers
            await self._notify_watchers(service_name, 'status_changed', service)
    
    async def start_monitoring(self):
        """Iniciar monitoreo de servicios"""
        
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Service monitoring started")
    
    async def stop_monitoring(self):
        """Detener monitoreo de servicios"""
        
        if self._health_check_task:
            self._health_check_task.cancel()
            
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        self.logger.info("Service monitoring stopped")
    
    def watch_service(self, service_name: str, callback: Callable):
        """Registrar watcher para cambios en un servicio"""
        
        if service_name not in self.service_watchers:
            self.service_watchers[service_name] = []
        
        self.service_watchers[service_name].append(callback)
    
    async def _register_in_redis(self, service: ServiceInstance):
        """Registrar servicio en Redis"""
        
        key = f"services:{service.service_name}:{service.instance_id}"
        
        service_data = {
            'service_name': service.service_name,
            'instance_id': service.instance_id,
            'host': service.host,
            'port': service.port,
            'health_check_url': service.health_check_url,
            'metadata': json.dumps(service.metadata),
            'status': service.status.value,
            'registration_time': service.registration_time.isoformat(),
            'version': service.version,
            'tags': json.dumps(list(service.tags))
        }
        
        await self.redis_client.hset(key, mapping=service_data)
        await self.redis_client.expire(key, 300)  # TTL de 5 minutos
        
        # Añadir a índice de servicios
        await self.redis_client.sadd(f"service_names", service.service_name)
        await self.redis_client.sadd(f"service_instances:{service.service_name}", service.instance_id)
    
    async def _deregister_from_redis(self, service_name: str, instance_id: str):
        """Desregistrar servicio de Redis"""
        
        key = f"services:{service_name}:{instance_id}"
        await self.redis_client.delete(key)
        await self.redis_client.srem(f"service_instances:{service_name}", instance_id)
    
    async def _discover_from_redis(self, service_name: str) -> List[ServiceInstance]:
        """Descubrir servicios desde Redis"""
        
        instance_ids = await self.redis_client.smembers(f"service_instances:{service_name}")
        services = []
        
        for instance_id in instance_ids:
            key = f"services:{service_name}:{instance_id}"
            service_data = await self.redis_client.hgetall(key)
            
            if service_data:
                service = ServiceInstance(
                    service_name=service_data['service_name'],
                    instance_id=service_data['instance_id'],
                    host=service_data['host'],
                    port=int(service_data['port']),
                    health_check_url=service_data['health_check_url'],
                    metadata=json.loads(service_data.get('metadata', '{}')),
                    status=ServiceStatus(service_data.get('status', 'available')),
                    version=service_data.get('version', 'unknown'),
                    tags=set(json.loads(service_data.get('tags', '[]')))
                )
                
                if service_data.get('registration_time'):
                    service.registration_time = datetime.fromisoformat(service_data['registration_time'])
                
                services.append(service)
        
        return services
    
    async def _update_in_redis(self, service: ServiceInstance):
        """Actualizar servicio en Redis"""
        
        key = f"services:{service.service_name}:{service.instance_id}"
        
        await self.redis_client.hset(key, mapping={
            'status': service.status.value,
            'metadata': json.dumps(service.metadata),
            'last_health_check': datetime.now().isoformat() if service.last_health_check else None
        })
    
    async def _health_check_loop(self):
        """Loop de health checks"""
        
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(10)
    
    async def _perform_health_checks(self):
        """Realizar health checks en todos los servicios"""
        
        services_to_check = list(self.local_registry.values())
        
        if self.redis_client:
            # También verificar servicios de Redis
            service_names = await self.redis_client.smembers("service_names")
            for service_name in service_names:
                redis_services = await self._discover_from_redis(service_name)
                services_to_check.extend(redis_services)
        
        # Ejecutar health checks en paralelo
        tasks = [self._check_service_health(service) for service in services_to_check]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service: ServiceInstance):
        """Verificar salud de un servicio individual"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    service.health_check_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    
                    if response.status == 200:
                        # Health check exitoso
                        service.consecutive_failures = 0
                        service.last_health_check = datetime.now()
                        
                        if service.status == ServiceStatus.STARTING:
                            await self.update_service_status(
                                service.service_name,
                                service.instance_id,
                                ServiceStatus.AVAILABLE
                            )
                    else:
                        await self._handle_health_check_failure(service)
                        
        except Exception as e:
            self.logger.warning(f"Health check failed for {service.service_key}: {e}")
            await self._handle_health_check_failure(service)
    
    async def _handle_health_check_failure(self, service: ServiceInstance):
        """Manejar fallo en health check"""
        
        service.consecutive_failures += 1
        
        if service.consecutive_failures >= self.failure_threshold:
            await self.update_service_status(
                service.service_name,
                service.instance_id,
                ServiceStatus.UNAVAILABLE
            )
    
    async def _cleanup_loop(self):
        """Loop de limpieza de servicios obsoletos"""
        
        while True:
            try:
                await self._cleanup_stale_services()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_stale_services(self):
        """Limpiar servicios obsoletos"""
        
        cutoff_time = datetime.now() - timedelta(minutes=10)
        stale_services = []
        
        for service in self.local_registry.values():
            if (service.last_health_check and 
                service.last_health_check < cutoff_time and
                service.status == ServiceStatus.UNAVAILABLE):
                stale_services.append(service)
        
        for service in stale_services:
            await self.deregister_service(service.service_name, service.instance_id)
            self.logger.info(f"Cleaned up stale service: {service.service_key}")
    
    async def _notify_watchers(self, service_name: str, event_type: str, service: ServiceInstance):
        """Notificar watchers de cambios"""
        
        watchers = self.service_watchers.get(service_name, [])
        
        for watcher in watchers:
            try:
                await watcher(event_type, service)
            except Exception as e:
                self.logger.error(f"Watcher notification failed: {e}")

class ServiceDiscoveryClient:
    """Cliente para service discovery"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.logger = logging.getLogger("service.discovery.client")
        
        # Cache local de servicios
        self.service_cache: Dict[str, List[ServiceInstance]] = {}
        self.cache_ttl = 60  # 1 minuto
        self.last_cache_update: Dict[str, datetime] = {}
    
    async def get_service_instance(self, 
                                 service_name: str,
                                 use_cache: bool = True) -> Optional[ServiceInstance]:
        """Obtener instancia de servicio con cache"""
        
        if use_cache and self._is_cache_valid(service_name):
            cached_services = self.service_cache.get(service_name, [])
            if cached_services:
                return cached_services[0]  # Devolver primera instancia disponible
        
        # Obtener de registry
        service = await self.registry.get_healthy_instance(service_name)
        
        if service and use_cache:
            self._update_cache(service_name, [service])
        
        return service
    
    async def get_all_service_instances(self, 
                                      service_name: str,
                                      use_cache: bool = True) -> List[ServiceInstance]:
        """Obtener todas las instancias de un servicio"""
        
        if use_cache and self._is_cache_valid(service_name):
            return self.service_cache.get(service_name, [])
        
        services = await self.registry.discover_services(
            service_name,
            status_filter=ServiceStatus.AVAILABLE
        )
        
        if use_cache:
            self._update_cache(service_name, services)
        
        return services
    
    def _is_cache_valid(self, service_name: str) -> bool:
        """Verificar si el cache es válido"""
        
        if service_name not in self.last_cache_update:
            return False
        
        last_update = self.last_cache_update[service_name]
        return (datetime.now() - last_update).total_seconds() < self.cache_ttl
    
    def _update_cache(self, service_name: str, services: List[ServiceInstance]):
        """Actualizar cache de servicios"""
        
        self.service_cache[service_name] = services
        self.last_cache_update[service_name] = datetime.now()
    
    def invalidate_cache(self, service_name: Optional[str] = None):
        """Invalidar cache"""
        
        if service_name:
            self.service_cache.pop(service_name, None)
            self.last_cache_update.pop(service_name, None)
        else:
            self.service_cache.clear()
            self.last_cache_update.clear()
```

### 3. Sistema Integrado de Health y Discovery

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
import signal
import sys

class HealthyLLMService:
    """Servicio LLM con health checks y service discovery integrados"""
    
    def __init__(self, 
                 service_name: str = "llm-service",
                 instance_id: str = None,
                 host: str = "localhost",
                 port: int = 8000):
        
        self.service_name = service_name
        self.instance_id = instance_id or f"{service_name}-{int(time.time())}"
        self.host = host
        self.port = port
        
        # Componentes
        self.health_checker = CompositeHealthChecker("llm-service")
        self.service_registry = ServiceRegistry()
        self.discovery_client = ServiceDiscoveryClient(self.service_registry)
        
        # Estado del servicio
        self.is_ready = False
        self.is_shutting_down = False
        
        # FastAPI app
        self.app = FastAPI(title=f"LLM Service - {self.instance_id}")
        self._setup_routes()
        
        self.logger = logging.getLogger(f"service.{self.instance_id}")
    
    def _setup_routes(self):
        """Configurar rutas de la API"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            
            if self.is_shutting_down:
                return JSONResponse(
                    status_code=503,
                    content={"status": "shutting_down", "message": "Service is shutting down"}
                )
            
            health_result = await self.health_checker.check_all()
            
            if health_result['overall_status'] == 'healthy':
                status_code = 200
            elif health_result['overall_status'] == 'degraded':
                status_code = 200  # Still serving traffic
            else:
                status_code = 503
            
            return JSONResponse(
                status_code=status_code,
                content=health_result
            )
        
        @self.app.get("/ready")
        async def readiness_check():
            """Readiness check endpoint"""
            
            if not self.is_ready or self.is_shutting_down:
                return JSONResponse(
                    status_code=503,
                    content={"ready": False, "message": "Service not ready"}
                )
            
            return {"ready": True, "message": "Service is ready"}
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Métricas del servicio"""
            
            health_result = await self.health_checker.check_all()
            
            return {
                "service_info": {
                    "name": self.service_name,
                    "instance_id": self.instance_id,
                    "uptime_seconds": time.time() - self.start_time if hasattr(self, 'start_time') else 0
                },
                "health": health_result['summary'],
                "discovery": {
                    "registered": self.instance_id in [s.instance_id for s in await self.service_registry.discover_services(self.service_name)]
                }
            }
        
        @self.app.post("/admin/drain")
        async def drain_service():
            """Drenar el servicio (graceful shutdown)"""
            
            await self.service_registry.update_service_status(
                self.service_name,
                self.instance_id,
                ServiceStatus.DRAINING
            )
            
            return {"message": "Service draining initiated"}
        
        @self.app.get("/discover/{service_name}")
        async def discover_service(service_name: str):
            """Descubrir otros servicios"""
            
            services = await self.discovery_client.get_all_service_instances(service_name)
            
            return {
                "service_name": service_name,
                "instances": [
                    {
                        "instance_id": s.instance_id,
                        "base_url": s.base_url,
                        "status": s.status.value,
                        "metadata": s.metadata
                    }
                    for s in services
                ]
            }
    
    async def start(self):
        """Iniciar el servicio"""
        
        self.start_time = time.time()
        
        # Configurar health checks
        await self._setup_health_checks()
        
        # Iniciar monitoreo de servicios
        await self.service_registry.start_monitoring()
        
        # Registrar el servicio
        service_instance = ServiceInstance(
            service_name=self.service_name,
            instance_id=self.instance_id,
            host=self.host,
            port=self.port,
            health_check_url=f"http://{self.host}:{self.port}/health",
            metadata={
                "start_time": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            version="1.0.0",
            tags={"llm", "api"}
        )
        
        await self.service_registry.register_service(service_instance)
        
        # Marcar como listo
        self.is_ready = True
        
        self.logger.info(f"Service {self.instance_id} started successfully")
        
        # Configurar signal handlers para graceful shutdown
        self._setup_signal_handlers()
    
    async def _setup_health_checks(self):
        """Configurar health checks del servicio"""
        
        # Health check de recursos del sistema
        system_check = SystemResourceHealthCheck("system_resources")
        self.health_checker.add_health_check(system_check, is_critical=True)
        
        # Health check de servicios externos (ejemplo)
        external_check = ExternalServiceHealthCheck(
            "openai_api", 
            "https://api.openai.com/v1/models"
        )
        self.health_checker.add_health_check(external_check, is_critical=False)
        
        # Aquí añadirías health checks para tus dependencias específicas
        # como base de datos, Redis, otros servicios LLM, etc.
    
    def _setup_signal_handlers(self):
        """Configurar manejadores de señales para graceful shutdown"""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown del servicio"""
        
        self.logger.info("Starting graceful shutdown")
        self.is_shutting_down = True
        
        # Cambiar estado a draining
        await self.service_registry.update_service_status(
            self.service_name,
            self.instance_id,
            ServiceStatus.DRAINING
        )
        
        # Esperar que drenen las conexiones existentes
        await asyncio.sleep(10)
        
        # Desregistrar servicio
        await self.service_registry.deregister_service(
            self.service_name,
            self.instance_id
        )
        
        # Detener monitoreo
        await self.service_registry.stop_monitoring()
        
        self.logger.info("Graceful shutdown completed")
        
        # Terminar proceso
        sys.exit(0)
    
    def run(self):
        """Ejecutar el servicio"""
        
        async def startup():
            await self.start()
        
        # Configurar uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        
        # Ejecutar startup y servidor
        loop = asyncio.get_event_loop()
        loop.run_until_complete(startup())
        loop.run_until_complete(server.serve())

# Ejemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    service = HealthyLLMService(
        service_name="llm-api",
        host="0.0.0.0",
        port=8000
    )
    
    service.run()
```

## 🎯 Mejores Prácticas

### 1. **Health Check Design**
- Checks rápidos y específicos
- Diferenciación entre liveness y readiness
- Timeouts apropiados
- Métricas de tendencia

### 2. **Service Discovery**
- TTL apropiados para registros
- Cache local con invalidación
- Fallback a configuración estática
- Balanceador de carga integrado

### 3. **Graceful Shutdown**
- Drenar conexiones existentes
- Desregistrar antes de cerrar
- Timeouts de shutdown configurables
- Logs detallados del proceso

### 4. **Monitoring Integration**
- Métricas de Prometheus
- Alertas en cambios de estado
- Dashboards de service topology
- SLI/SLO tracking

## 📊 Métricas de Observabilidad

- **Service Availability**: Porcentaje de tiempo activo
- **Health Check Duration**: Tiempo de respuesta de checks
- **Service Discovery Latency**: Tiempo de resolución
- **Registration/Deregistration Rate**: Frecuencia de cambios
- **Circuit Breaker State**: Estado de protecciones

## 🔧 Próximo Paso

En el **Laboratorio 7** implementaremos una arquitectura resiliente completa que integre todos estos patrones en un sistema LLM productivo.

## 📖 Recursos Adicionales

- [Health Check Patterns](https://microservices.io/patterns/observability/health-check-api.html)
- [Service Discovery Patterns](https://microservices.io/patterns/service-registry.html)
- [Graceful Shutdown](https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-terminating-with-grace)
- [Spring Boot Actuator](https://docs.spring.io/spring-boot/docs/current/reference/html/actuator.html)
