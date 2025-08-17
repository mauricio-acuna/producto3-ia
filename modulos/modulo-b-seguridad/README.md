# Módulo B - Seguridad y Privacidad

## 🎯 Objetivos de Aprendizaje

- Identificar y mitigar ataques de prompt injection
- Implementar sanitización de inputs
- Gestionar secretos y credenciales de forma segura
- Cumplir con regulaciones de privacidad (GDPR, PII)

## 📚 Contenido

### 1. Amenazas de Seguridad en LLMs
- Prompt injection y sus variantes
- Data poisoning
- Model extraction
- Jailbreaking

### 2. Sanitización y Validación de Inputs
- Filtros de contenido
- Detección de patrones maliciosos
- Validación semántica

### 3. Gestión de Secretos
- Vault de secretos
- Rotación de API keys
- Principio de menor privilegio

### 4. Privacidad y Compliance
- Detección de PII (Personally Identifiable Information)
- Cumplimiento GDPR
- Anonimización de datos
- Logs seguros

### 5. Guardrails de Seguridad
- Implementación de safety filters
- Rate limiting
- Circuit breakers para seguridad

## 🛠️ Herramientas

- LangChain Safety
- Microsoft Presidio (PII detection)
- HashiCorp Vault
- OWASP ZAP (testing)

## 📝 Laboratorios

1. **Lab 1**: Detectar y bloquear prompt injection
2. **Lab 2**: Implementar detección de PII
3. **Lab 3**: Configurar gestión segura de secretos
4. **Lab 4**: Crear pipeline de security testing

## ✅ Criterios de Evaluación

- [ ] Sistema de detección de prompt injection ≥80% precisión
- [ ] Gestión segura de API keys implementada
- [ ] Detección de PII funcionando
- [ ] Logs sin información sensible
- [ ] Safety filters configurados

## 🚨 Checklist de Seguridad

- [ ] Inputs validados y sanitizados
- [ ] Secretos no hardcodeados
- [ ] Logs sin PII
- [ ] Rate limiting implementado
- [ ] Monitoreo de anomalías activo

## 📖 Recursos Adicionales

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Security Guidelines](https://www.nist.gov/itl/ai-risk-management-framework)
- Templates de seguridad en `/templates/security/`
