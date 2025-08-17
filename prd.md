
# 📄 PRD — Portal 3 “LLMOps: Operación, Seguridad y CI/CD”

## 1. Introducción

### 1.1 Propósito

El **Portal 3** enseña cómo pasar de un prototipo de agente/RAG a un sistema que puede funcionar de manera **segura, observable y sostenible en producción**.
El foco está en **LLMOps**: monitoreo de métricas clave, seguridad y privacidad, compliance (GDPR/PII), control de costes, resiliencia y pipelines de CI/CD adaptados a IA.

### 1.2 Alcance

Este portal cubre:

* **Observabilidad:** OpenTelemetry, métricas de latencia, coste, precisión.
* **Seguridad:** prompt injection, sanitización de inputs, gestión de secretos.
* **Cumplimiento:** privacidad de datos (PII, GDPR, PCI).
* **Costes y rendimiento:** presupuestos, optimización tokens, caching.
* **Resiliencia:** fallback de modelos, circuit breakers, timeouts.
* **CI/CD:** integración de evals y gates automáticos en pipelines.

No incluye: optimizaciones profundas de hardware (se abordan en Portal 4).

---

## 2. Público objetivo y usuarios

### 2.1 Perfil primario

* Devs que ya construyeron agentes/RAG en Portal 2.
* **Ingenieros mid/senior** que buscan llevar prototipos a producción segura.
* Perfíl típico: *Backend dev*, *DevOps*, *MLOps engineer* en transición a *AI engineer*.

### 2.2 Problemas a resolver

* Despliegues inseguros (prompt injection, fuga de secretos).
* Costes descontrolados en APIs de modelos.
* Ausencia de métricas y alertas.
* Pipelines de CI/CD sin gates de calidad.

---

## 3. Objetivos y métricas de éxito

### 3.1 Objetivos de producto

1. Enseñar a instrumentar un agente con **OpenTelemetry** (spans, métricas, logs).
2. Implementar **guardrails de seguridad y privacidad**.
3. Configurar **presupuestos de coste** y aplicar técnicas de optimización.
4. Garantizar **resiliencia operacional** (fallback de modelos, circuit breakers).
5. Crear un **pipeline CI/CD** que ejecute evals y bloquee despliegues inseguros.

### 3.2 KPIs / métricas

* % de alumnos que implementan OTel en su capstone ≥ 50%.
* Reducción del coste promedio en prompts ≥ 20% aplicando técnicas vistas.
* Tasa de detección de prompt injection ≥ 80% en tests.
* % de alumnos que configuran un pipeline CI/CD con gates ≥ 60%.

---

## 4. Requisitos funcionales

### 4.1 Currículo

* **Módulo A — Observabilidad**
  *OTel, métricas clave, dashboards (Grafana/Prometheus).*
* **Módulo B — Seguridad y privacidad**
  *Prompt injection, sanitización, gestión de secretos, PII/GDPR.*
* **Módulo C — Costes y rendimiento**
  *Optimización de tokens, caching, batching, selección de modelo.*
* **Módulo D — Resiliencia operacional**
  *Circuit breakers, fallbacks, timeouts, degradación controlada.*
* **Módulo E — CI/CD para agentes**
  *Pipelines con gates de calidad, evals automáticas, integración en GitHub Actions/GitLab CI.*
* **Capstone:** desplegar un agente con RAG híbrido en entorno CI/CD, instrumentado con OTel, con seguridad básica y costes bajo control.

### 4.2 Funcionalidades del portal

* **Playbooks prácticos:** recetas para instrumentar, asegurar y optimizar.
* **Dashboards preconfigurados:** ejemplos de métricas en Grafana.
* **Templates CI/CD:** YAML listos para adaptar.
* **Laboratorios:** simular ataques de prompt injection, medir coste en tokens, configurar gates.

---

## 5. Requisitos no funcionales

### 5.1 UX

* Visualizaciones de métricas y dashboards.
* Flujos paso a paso con ejemplos reales.
* Checklist de seguridad por módulo.

### 5.2 SEO

* Keywords: “LLMOps”, “seguridad agentes IA”, “observabilidad LLM”, “CI/CD para IA”.
* Schema.org: `Course`, `HowTo`, `SoftwareApplication`.

### 5.3 Performance

* Recursos gráficos optimizados (ej. capturas de dashboards).
* Ejemplos de pipelines ligeros (no más de 3–5 min en CI).

---

## 6. Roadmap de desarrollo de portal

| Semana | Entregable                                             |
| ------ | ------------------------------------------------------ |
| 1      | Landing + Módulo A (Observabilidad)                    |
| 2      | Módulo B (Seguridad) + laboratorio de prompt injection |
| 3      | Módulo C (Costes)                                      |
| 4      | Módulo D (Resiliencia)                                 |
| 5      | Módulo E (CI/CD)                                       |
| 6      | Capstone final + rúbrica                               |

---

## 7. Recursos de aprendizaje incluidos

* **Plantillas:** `safety.yaml`, config de OTel, pipelines CI/CD (`.github/workflows/ai-ci.yaml`).
* **Cheat-sheets:** métricas clave, top técnicas de optimización de costes.
* **Datasets de ataque:** ejemplos de prompt injection y PII.
* **Ejemplo de dashboard Grafana** exportable.

---

## 8. Entregables para el alumno

* Repo con agente instrumentado con OTel.
* Configuración de seguridad (safety.yaml, sanitización).
* Reporte de costes/latencia/precisión.
* Pipeline CI/CD con evals automáticas.
* Demo o grabación mostrando fallback/resiliencia.

---

## 9. Glosario (extracto)

* **OTel (OpenTelemetry):** estándar abierto para observabilidad.
* **Prompt injection:** ataque que manipula el input para romper seguridad.
* **Circuit breaker:** patrón que corta llamadas a un servicio cuando falla en exceso.
* **Eval gate:** prueba automática en CI/CD que bloquea despliegues si no se cumplen métricas mínimas.

---

## 10. Riesgos y mitigaciones

* **Riesgo:** complejidad de OTel y dashboards.

  * **Mitigación:** plantillas + demo preconfigurado.
* **Riesgo:** dificultad de CI/CD en alumnos sin experiencia DevOps.

  * **Mitigación:** YAMLs listos para copiar/pegar.
* **Riesgo:** sobrecarga de temas (seguridad, coste, observabilidad).

  * **Mitigación:** laboratorios cortos y progresivos.

---

## 11. KPI de seguimiento interno

* % de alumnos con OTel funcionando.
* % de pipelines CI/CD completados.
* Precisión promedio en evals de seguridad.
* Ratio de reducción de coste en capstone.

---

## 12. Cierre

Este portal transforma a los alumnos en **ingenieros capaces de operar agentes en entornos productivos**, con métricas, seguridad y pipelines sólidos. Quien lo complete podrá postularse a roles de **AI Ops Engineer / Applied AI DevOps** y tener ventajas frente a devs que sólo saben prototipar.

---
