# Módulo D - Resiliencia Operacional

## 🎯 Objetivos de Aprendizaje

- Implementar circuit breakers para servicios LLM
- Configurar fallbacks y degradación controlada
- Establecer timeouts apropiados
- Diseñar sistemas tolerantes a fallos

## 📚 Contenido

### 1. Patrones de Resiliencia
- Circuit Breaker pattern
- Retry with exponential backoff
- Bulkhead isolation
- Timeout patterns

### 2. Fallbacks Inteligentes
- Model fallbacks (GPT-4 → GPT-3.5)
- Response caching como fallback
- Respuestas predeterminadas
- Degradación graceful

### 3. Monitoreo de Salud
- Health checks
- SLA monitoring
- Error rate tracking
- Latency percentiles

### 4. Recuperación Automática
- Auto-scaling
- Load balancing
- Failover mechanisms
- Recovery strategies

### 5. Testing de Resiliencia
- Chaos engineering
- Fault injection
- Load testing
- Disaster recovery drills

## 🛠️ Herramientas

- Hystrix/Resilience4j (Circuit breakers)
- Chaos Monkey
- Load testing tools (Locust, JMeter)
- Health check libraries

## 📝 Laboratorios

1. **Lab 1**: Implementar circuit breaker básico
2. **Lab 2**: Configurar fallbacks de modelo
3. **Lab 3**: Setup de monitoring de salud
4. **Lab 4**: Simular fallas y recuperación

## ✅ Criterios de Evaluación

- [ ] Circuit breakers funcionando correctamente
- [ ] Fallbacks de modelo configurados
- [ ] Timeouts apropiados establecidos
- [ ] Health checks implementados
- [ ] Recovery automático funcionando
- [ ] Tests de resiliencia ejecutados

## 🔧 Patrones de Implementación

### Circuit Breaker
```python
# Ejemplo de configuración
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,
    "timeout": 60,
    "expected_exception": OpenAIError
}
```

### Fallback Strategy
1. **Modelo Principal** (GPT-4)
2. **Modelo Secundario** (GPT-3.5-turbo)
3. **Cache de Respuestas**
4. **Respuesta Default**

### Health Metrics
- **Availability**: % uptime del servicio
- **Error Rate**: % de requests que fallan
- **Response Time**: latencia promedio
- **Throughput**: requests por segundo

## 🚨 Alertas Críticas

- Circuit breaker activado
- Error rate > 5%
- Latencia > P95 threshold
- Modelo principal no disponible

## 📖 Recursos Adicionales

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Chaos Engineering Principles](https://principlesofchaos.org/)
- Templates de resiliencia en `/templates/resilience/`
