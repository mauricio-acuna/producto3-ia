# 📊 Laboratorio 2: Dashboards y Alertas con Grafana

## 🎯 Objetivos del Laboratorio

- Crear dashboards personalizados en Grafana para métricas de LLM
- Configurar alertas inteligentes para métricas críticas
- Implementar SLIs (Service Level Indicators) y SLOs (Service Level Objectives)
- Optimizar visualizaciones para operaciones LLMOps

## ⏱️ Tiempo Estimado: 60 minutos

## 📋 Prerrequisitos

- Laboratorio 1 completado (agente instrumentado funcionando)
- Grafana, Prometheus y Jaeger ejecutándose
- Datos de telemetría siendo generados

## 🎨 Paso 1: Configuración Inicial de Grafana

### 1.1 Acceder a Grafana

1. Abrir http://localhost:3000
2. Login: `admin` / `admin123`
3. Verificar datasource de Prometheus configurado

### 1.2 Configurar Datasources

Crear archivo `config/grafana/datasources/datasources.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: 5s
      queryTimeout: 60s

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
    jsonData:
      tracesToLogs:
        datasourceUid: loki
        tags: ['job', 'instance', 'pod', 'namespace']
```

### 1.3 Configurar Dashboard Provider

Crear `config/grafana/dashboards/dashboards.yml`:

```yaml
apiVersion: 1

providers:
  - name: 'LLMOps Dashboards'
    orgId: 1
    folder: 'LLMOps'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

## 📊 Paso 2: Dashboard Principal de LLMOps

### 2.1 Crear Dashboard LLMOps Overview

Crear `config/grafana/dashboards/llmops-overview.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "LLMOps Overview - Lab 2",
    "tags": ["llmops", "openai", "observability"],
    "style": "dark",
    "timezone": "browser",
    "refresh": "5s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "LLM Requests Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])",
            "legendFormat": "{{model}} ({{status}})",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "palette-classic"
            },
            "custom": {
              "displayMode": "basic",
              "orientation": "auto"
            },
            "mappings": [],
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "red", "value": 10}
              ]
            },
            "unit": "reqps"
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Average Response Time",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(llm_request_duration_seconds_sum[5m]) / rate(llm_request_duration_seconds_count[5m])",
            "legendFormat": "Avg Response Time",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "thresholds"},
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 2},
                {"color": "red", "value": 5}
              ]
            },
            "unit": "s"
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "Token Usage Rate",
        "type": "stat", 
        "targets": [
          {
            "expr": "rate(llm_tokens_used_total[5m])",
            "legendFormat": "{{type}} tokens/sec",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "decimals": 0
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "title": "Cost Rate (USD/hour)",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(llm_cost_usd_total[5m]) * 3600",
            "legendFormat": "Cost/hour",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "decimals": 4,
            "color": {"mode": "thresholds"},
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0}
      },
      {
        "id": 5,
        "title": "Request Success Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(llm_requests_total{status=\"success\"}[5m]) / rate(llm_requests_total[5m]) * 100",
            "legendFormat": "Success Rate %",
            "refId": "A"
          },
          {
            "expr": "rate(llm_requests_total{status=\"error\"}[5m]) / rate(llm_requests_total[5m]) * 100",
            "legendFormat": "Error Rate %",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "linear",
              "fillOpacity": 20
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 6,
        "title": "Response Time Distribution",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99",
            "refId": "C"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "linear"
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      },
      {
        "id": 7,
        "title": "Model Usage Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (model) (rate(llm_requests_total[5m]))",
            "legendFormat": "{{model}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "custom": {
              "displayLabels": ["name", "percent"]
            }
          }
        },
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16}
      },
      {
        "id": 8,
        "title": "Token Efficiency (Tokens/Request)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(llm_tokens_used_total{type=\"prompt\"}[5m]) / rate(llm_requests_total[5m])",
            "legendFormat": "Avg Prompt Tokens",
            "refId": "A"
          },
          {
            "expr": "rate(llm_tokens_used_total{type=\"completion\"}[5m]) / rate(llm_requests_total[5m])",
            "legendFormat": "Avg Completion Tokens",
            "refId": "B"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short"
          }
        },
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 16}
      },
      {
        "id": 9,
        "title": "Conversation Metrics",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(agent_conversations_total[5m])",
            "legendFormat": "New Conversations/sec",
            "refId": "A"
          },
          {
            "expr": "rate(agent_messages_total[5m])",
            "legendFormat": "Messages/sec",
            "refId": "B"
          }
        ],
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 16}
      }
    ]
  }
}
```

## 🚨 Paso 3: Configurar Alertas

### 3.1 Reglas de Alertas en Prometheus

Crear `config/prometheus-rules.yml`:

```yaml
groups:
  - name: llmops-alerts
    rules:
      # Alta latencia
      - alert: HighLLMLatency
        expr: histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m])) > 5
        for: 2m
        labels:
          severity: warning
          service: llm-agent
        annotations:
          summary: "High LLM response latency detected"
          description: "95th percentile latency is {{ $value }}s for the last 5 minutes"

      # Alto rate de errores
      - alert: HighLLMErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
          service: llm-agent
        annotations:
          summary: "High LLM error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

      # Alto consumo de tokens
      - alert: HighTokenUsage
        expr: rate(llm_tokens_used_total[5m]) > 10000
        for: 5m
        labels:
          severity: warning
          service: llm-agent
        annotations:
          summary: "High token usage detected"
          description: "Token usage rate is {{ $value }} tokens/sec for the last 5 minutes"

      # Alto coste
      - alert: HighLLMCost
        expr: rate(llm_cost_usd_total[5m]) * 3600 > 10
        for: 3m
        labels:
          severity: warning
          service: llm-agent
        annotations:
          summary: "High LLM cost detected"
          description: "Current cost rate is ${{ $value }}/hour"

      # Servicio no disponible
      - alert: LLMServiceDown
        expr: up{job="llm-agent"} == 0
        for: 1m
        labels:
          severity: critical
          service: llm-agent
        annotations:
          summary: "LLM service is down"
          description: "LLM service has been down for more than 1 minute"
```

### 3.2 Actualizar Prometheus Config

Modificar `config/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "prometheus-rules.yml"

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'llm-agent'
    static_configs:
      - targets: ['host.docker.internal:8000']
    scrape_interval: 10s
    metrics_path: /metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 3.3 Configurar Grafana Alerts

En Grafana, crear alertas:

1. **Dashboard → Panel → Alert Tab**
2. **Configurar condiciones**:

```json
{
  "conditions": [
    {
      "evaluator": {
        "params": [5],
        "type": "gt"
      },
      "operator": {
        "type": "and"
      },
      "query": {
        "params": ["A", "5m", "now"]
      },
      "reducer": {
        "params": [],
        "type": "avg"
      },
      "type": "query"
    }
  ],
  "executionErrorState": "alerting",
  "frequency": "10s",
  "handler": 1,
  "name": "High Response Time Alert",
  "noDataState": "no_data",
  "notifications": []
}
```

## 📈 Paso 4: SLIs y SLOs

### 4.1 Definir SLIs (Service Level Indicators)

Crear dashboard `SLI-SLO-Dashboard.json`:

```json
{
  "dashboard": {
    "title": "SLI/SLO Dashboard",
    "panels": [
      {
        "title": "Availability SLI",
        "type": "stat",
        "targets": [
          {
            "expr": "avg_over_time((rate(llm_requests_total{status=\"success\"}[5m]) / rate(llm_requests_total[5m]))[24h:5m]) * 100",
            "legendFormat": "24h Availability"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 99},
                {"color": "green", "value": 99.9}
              ]
            }
          }
        }
      },
      {
        "title": "Latency SLI (P95 < 3s)",
        "type": "stat",
        "targets": [
          {
            "expr": "(sum(rate(llm_request_duration_seconds_bucket{le=\"3\"}[5m])) / sum(rate(llm_request_duration_seconds_count[5m]))) * 100",
            "legendFormat": "P95 < 3s %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 95},
                {"color": "green", "value": 99}
              ]
            }
          }
        }
      },
      {
        "title": "Cost Efficiency SLI",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(llm_cost_usd_total[5m]) / rate(llm_requests_total[5m])",
            "legendFormat": "Cost per Request"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 0.01},
                {"color": "red", "value": 0.05}
              ]
            }
          }
        }
      }
    ]
  }
}
```

### 4.2 Definir SLOs

Crear archivo `slo-definitions.md`:

```markdown
# Service Level Objectives (SLOs)

## Availability SLO
- **Target**: 99.5% availability over 30 days
- **Measurement**: Successful requests / Total requests
- **Error Budget**: 0.5% (3.6 hours downtime per month)

## Latency SLO  
- **Target**: 95% of requests complete within 3 seconds
- **Measurement**: P95 latency < 3s
- **Error Budget**: 5% of requests can exceed 3s

## Cost Efficiency SLO
- **Target**: Average cost per request < $0.02
- **Measurement**: Total cost / Total requests
- **Error Budget**: 10% variance allowed

## Token Efficiency SLO
- **Target**: Average tokens per request < 1000
- **Measurement**: Total tokens / Total requests
- **Error Budget**: 20% variance allowed
```

## 🧪 Paso 5: Testing y Validación

### 5.1 Script de Carga de Trabajo

Crear `load_test.py`:

```python
import asyncio
import aiohttp
import random
import time
from typing import List

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.messages = [
            "¿Qué es OpenTelemetry?",
            "Explica los beneficios de la observabilidad",
            "¿Cómo configurar Grafana?",
            "¿Qué son las métricas de SLI/SLO?",
            "Diferencias entre traces y logs",
            "¿Cómo optimizar costes de LLM?",
            "Explica el circuit breaker pattern",
            "¿Qué es Prometheus?",
            "Ventajas de usar dashboards",
            "¿Cómo funciona el OTLP?"
        ]
    
    async def send_message(self, session: aiohttp.ClientSession, message: str):
        """Enviar un mensaje al agente"""
        try:
            async with session.post(
                f"{self.base_url}/chat",
                json={"message": message, "model": "gpt-3.5-turbo"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def run_load_test(self, concurrent_users: int = 5, duration_minutes: int = 5):
        """Ejecutar test de carga"""
        print(f"Starting load test: {concurrent_users} users for {duration_minutes} minutes")
        
        end_time = time.time() + (duration_minutes * 60)
        results = {"success": 0, "errors": 0}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            while time.time() < end_time:
                # Crear tareas concurrentes
                for _ in range(concurrent_users):
                    message = random.choice(self.messages)
                    task = asyncio.create_task(self.send_message(session, message))
                    tasks.append(task)
                
                # Ejecutar batch de requests
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Procesar resultados
                for result in batch_results:
                    if isinstance(result, dict) and result.get("success"):
                        results["success"] += 1
                    else:
                        results["errors"] += 1
                
                tasks.clear()
                
                # Esperar antes del siguiente batch
                await asyncio.sleep(random.uniform(1, 3))
                
                # Mostrar progreso cada 30 segundos
                if int(time.time()) % 30 == 0:
                    total = results["success"] + results["errors"]
                    error_rate = (results["errors"] / total * 100) if total > 0 else 0
                    print(f"Progress: {total} requests, {error_rate:.1f}% error rate")
        
        # Resultados finales
        total = results["success"] + results["errors"]
        print(f"\nLoad test completed:")
        print(f"Total requests: {total}")
        print(f"Successful: {results['success']}")
        print(f"Errors: {results['errors']}")
        print(f"Error rate: {results['errors'] / total * 100:.2f}%")

if __name__ == "__main__":
    tester = LoadTester()
    asyncio.run(tester.run_load_test(concurrent_users=3, duration_minutes=2))
```

### 5.2 Ejecutar Test de Carga

```bash
python load_test.py
```

Mientras se ejecuta, observar:
1. **Grafana dashboards** actualizándose en tiempo real
2. **Métricas de latencia** incrementándose
3. **Uso de tokens** y costes
4. **Posibles alertas** disparándose

## ✅ Criterios de Evaluación

### Básicos:
- [ ] Dashboard principal funcionando con todas las métricas
- [ ] Al menos 3 alertas configuradas correctamente
- [ ] SLIs definidos y visibles
- [ ] Test de carga genera telemetría visible

### Intermedios:
- [ ] Alertas se disparan correctamente bajo condiciones específicas
- [ ] SLOs documentados con error budgets
- [ ] Dashboards optimizados para diferentes audiencias
- [ ] Métricas de negocio incluidas

### Avanzados:
- [ ] Alertas integradas con sistemas de notificación
- [ ] SLO burn rate alerts configuradas
- [ ] Dashboards responsivos y performance optimizados
- [ ] Métricas de capacity planning implementadas

## 🎯 Ejercicios Adicionales

1. **Crear dashboard específico por modelo**: Comparar rendimiento entre GPT-3.5 y GPT-4
2. **Implementar alertas de burn rate**: Alertas cuando el error budget se consume muy rápido
3. **Dashboard de costes**: Tracking detallado de gastos por usuario/sesión
4. **Métricas de calidad**: Implementar scoring de respuestas y añadir al dashboard

## 🎉 ¡Excelente Trabajo!

Has creado un sistema completo de observabilidad para LLMOps con:
- ✅ Dashboards informativos y actionables
- ✅ Alertas inteligentes para métricas críticas  
- ✅ SLIs/SLOs para gestión de calidad de servicio
- ✅ Testing automatizado para validación

En el próximo módulo nos enfocaremos en **Seguridad y Privacidad** para proteger nuestros agentes contra amenazas comunes.
