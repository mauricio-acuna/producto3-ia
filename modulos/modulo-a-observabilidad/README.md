# Módulo A - Observabilidad

## 🎯 Objetivos de Aprendizaje

- Implementar OpenTelemetry (OTel) para instrumentar agentes IA
- Configurar métricas clave de rendimiento
- Crear dashboards con Grafana/Prometheus
- Establecer alertas operacionales

## 📚 Contenido

### 1. Introducción a OpenTelemetry
- Conceptos fundamentales: traces, spans, métricas, logs
- Arquitectura de observabilidad para LLMs

### 2. Instrumentación de Agentes
- Setup de OTel para aplicaciones Python
- Instrumentación automática vs manual
- Contexto distribuido en llamadas a LLMs

### 3. Métricas Clave para LLMOps
- **Latencia**: tiempo de respuesta, TTFT (Time to First Token)
- **Coste**: tokens consumidos, precio por request
- **Calidad**: precisión, relevancia de respuestas
- **Disponibilidad**: uptime, error rates

### 4. Dashboards y Visualización
- Configuración de Grafana
- Dashboards preconfigurados
- Alertas y notificaciones

## 🛠️ Herramientas

- OpenTelemetry SDK
- Grafana
- Prometheus
- Jaeger (opcional para tracing)

## 📝 Laboratorios

1. **Lab 1**: Instrumentar un agente simple con OTel
2. **Lab 2**: Configurar métricas personalizadas
3. **Lab 3**: Crear dashboard de monitoreo

## ✅ Criterios de Evaluación

- [ ] Agente instrumentado con OTel funcionando
- [ ] Métricas de latencia, coste y calidad capturadas
- [ ] Dashboard básico configurado
- [ ] Alertas configuradas para métricas críticas

## 📖 Recursos Adicionales

- [OpenTelemetry Documentation](https://opentelemetry.io/)
- [Grafana Tutorials](https://grafana.com/tutorials/)
- Plantillas de configuración en `/templates/observability/`
