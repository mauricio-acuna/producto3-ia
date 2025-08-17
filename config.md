# Configuración del Proyecto Portal 3

## Información del Proyecto
PROJECT_NAME="Portal 3 - LLMOps"
PROJECT_VERSION="1.0.0"
PROJECT_DESCRIPTION="Curso de LLMOps: Operación, Seguridad y CI/CD para Agentes IA"

## Roadmap de Desarrollo
CURRENT_WEEK=3
CURRENT_DELIVERABLE="Módulo C (Optimización) - Gestión de costos y performance"

## Progreso Completado ✅
- ✅ Semana 1: Módulo A (Observabilidad) - COMPLETADO
  - Lección 1: Introducción a OpenTelemetry para LLMs
  - Laboratorio 1: Instrumentación básica con OTel
  - Laboratorio 2: Dashboards y alertas con Grafana
  - Ejemplos avanzados de instrumentación

- ✅ Semana 2: Módulo B (Seguridad) - COMPLETADO
  - ✅ Lección 1: Detección y Prevención de Prompt Injection
  - ✅ Laboratorio 3: Detectar y Bloquear Prompt Injection
  - ✅ Lección 2: PII Detection y GDPR Compliance
  - ✅ Laboratorio 4: Sistema de Compliance con Microsoft Presidio

- 🚧 Semana 3: Módulo C (Optimización) - INICIANDO
  - 🚧 Lección 1: Optimización de Costos en LLMs
  - 🚧 Laboratorio 5: Implementar Caching y Token Optimization
  - 🚧 Lección 2: Performance y Escalabilidad
  - 🚧 Laboratorio 6: Load Balancing y Rate Limiting

## Configuración de Desarrollo
PYTHON_VERSION="3.9+"
NODE_VERSION="16+" # Para herramientas frontend si es necesario

## Tecnologías Principales
LLM_PROVIDERS=["OpenAI", "Anthropic", "Azure OpenAI"]
OBSERVABILITY_STACK=["OpenTelemetry", "Grafana", "Prometheus"]
SECURITY_TOOLS=["LangChain Safety", "Microsoft Presidio", "HashiCorp Vault"]
VECTOR_DATABASES=["Pinecone", "Weaviate", "Chroma"]
CACHE_SOLUTIONS=["Redis", "Memcached"]

## Métricas de Éxito (Targets)
OTEL_IMPLEMENTATION_TARGET=50  # % de alumnos
COST_REDUCTION_TARGET=20       # % reducción de costes
PROMPT_INJECTION_DETECTION=80  # % precisión
CICD_PIPELINE_TARGET=60        # % de alumnos

## URLs y Enlaces
GITHUB_REPO="https://github.com/mauricio-acuna/producto3-ia.git"
DOCUMENTATION_BASE="/docs"

## Estructura de Commits
# Usar conventional commits:
# feat: nueva funcionalidad
# fix: corrección de bugs
# docs: cambios en documentación
# style: formateo, espacios en blanco
# refactor: refactoring de código
# test: agregar o corregir tests
# chore: tareas de mantenimiento
