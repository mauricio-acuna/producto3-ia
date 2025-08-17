# 🔒 Lección 2: Detección de PII y Compliance con GDPR

## 🎯 Objetivos de la Lección

Al finalizar esta lección, serás capaz de:
- Identificar y detectar información personal identificable (PII)
- Implementar sistemas de anonimización automática
- Configurar compliance con GDPR, CCPA y otras regulaciones
- Gestionar consentimiento y derechos de los usuarios
- Crear pipelines de data governance para LLMs

## 📊 ¿Qué es PII (Personally Identifiable Information)?

**PII** es cualquier información que puede usarse para identificar a una persona específica, ya sea directa o indirectamente.

### Tipos de PII

#### 1. PII Directo (Direct PII)
```python
DIRECT_PII_TYPES = {
    'nombres_completos': ['John Smith', 'María García López'],
    'numeros_identificacion': ['123-45-6789', 'DNI 12345678-A'],
    'emails': ['john@company.com', 'maria.garcia@email.es'],
    'telefonos': ['+1-555-123-4567', '+34-666-123-456'],
    'direcciones': ['123 Main St, New York, NY 10001']
}
```

#### 2. PII Indirecto (Quasi-identifiers)
```python
INDIRECT_PII_TYPES = {
    'demograficos': ['edad: 34', 'género: masculino'],
    'geograficos': ['código postal: 10001', 'barrio: Manhattan'],
    'profesionales': ['empresa: TechCorp', 'cargo: Senior Developer'],
    'temporales': ['fecha nacimiento: 1989-03-15']
}
```

#### 3. PII Sensible (Sensitive PII)
```python
SENSITIVE_PII_TYPES = {
    'financiero': ['4532-1234-5678-9012', 'IBAN: ES91 2100 0418 4502 0005 1332'],
    'medico': ['diabetes tipo 2', 'alergia a penicilina'],
    'biometrico': ['huella dactilar', 'reconocimiento facial'],
    'legal': ['antecedentes penales', 'estado civil']
}
```

## 🔍 Implementación de Detección de PII

### 1. Detector Básico con Regex

```python
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class PIIType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IP_ADDRESS = "ip_address"
    PERSON_NAME = "person_name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"

@dataclass
class PIIMatch:
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float
    context: str

class RegexPIIDetector:
    """Detector de PII basado en expresiones regulares"""
    
    def __init__(self):
        self.patterns = {
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIIType.PHONE: [
                r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',  # US
                r'\+?34[-.\s]?[6-9][0-9]{2}[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}',   # España
                r'\+?[1-9]\d{1,14}'  # International format
            ],
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                r'\b\d{9}\b'  # XXXXXXXXX
            ],
            PIIType.CREDIT_CARD: [
                r'\b4[0-9]{12}(?:[0-9]{3})?\b',  # Visa
                r'\b5[1-5][0-9]{14}\b',  # MasterCard
                r'\b3[47][0-9]{13}\b',  # American Express
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'
            ],
            PIIType.IBAN: [
                r'\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b'
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IPv4
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'  # IPv6
            ],
            PIIType.DATE_OF_BIRTH: [
                r'\b(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[012])[-/](?:19|20)\d\d\b',
                r'\b(?:19|20)\d\d[-/](?:0[1-9]|1[012])[-/](?:0[1-9]|[12][0-9]|3[01])\b'
            ],
            PIIType.PASSPORT: [
                r'\b[A-Z]{1,2}[0-9]{6,9}\b'  # Formato general
            ]
        }
        
        # Compilar patrones para mejor rendimiento
        self.compiled_patterns = {}
        for pii_type, patterns in self.patterns.items():
            self.compiled_patterns[pii_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect(self, text: str) -> List[PIIMatch]:
        """Detectar PII en el texto usando regex"""
        matches = []
        
        for pii_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    # Extraer contexto (50 caracteres antes y después)
                    start_context = max(0, match.start() - 50)
                    end_context = min(len(text), match.end() + 50)
                    context = text[start_context:end_context]
                    
                    # Calcular confianza basada en el tipo
                    confidence = self._calculate_confidence(pii_type, match.group())
                    
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        context=context
                    ))
        
        return self._deduplicate_matches(matches)
    
    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """Calcular confianza del match basado en validaciones adicionales"""
        
        if pii_type == PIIType.EMAIL:
            # Validar formato de email más estricto
            if '@' in value and '.' in value.split('@')[1]:
                return 0.95
            return 0.7
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Validar usando algoritmo de Luhn
            return 0.9 if self._luhn_check(value) else 0.6
        
        elif pii_type == PIIType.PHONE:
            # Validar longitud y formato
            digits_only = re.sub(r'[^\d]', '', value)
            if 7 <= len(digits_only) <= 15:
                return 0.85
            return 0.6
        
        elif pii_type == PIIType.SSN:
            # Validar que no sean todos ceros o números secuenciales
            digits_only = re.sub(r'[^\d]', '', value)
            if digits_only == '000000000' or digits_only == '123456789':
                return 0.3
            return 0.9
        
        return 0.8  # Confianza por defecto
    
    def _luhn_check(self, card_number: str) -> bool:
        """Implementar algoritmo de Luhn para validar números de tarjeta"""
        digits = [int(d) for d in re.sub(r'[^\d]', '', card_number)]
        
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        return sum(digits) % 10 == 0
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Eliminar matches duplicados o superpuestos"""
        if not matches:
            return matches
        
        # Ordenar por posición
        matches.sort(key=lambda x: (x.start, x.end))
        
        deduplicated = []
        for match in matches:
            # Verificar si se superpone con el último match añadido
            if not deduplicated or match.start >= deduplicated[-1].end:
                deduplicated.append(match)
            else:
                # Mantener el match con mayor confianza
                if match.confidence > deduplicated[-1].confidence:
                    deduplicated[-1] = match
        
        return deduplicated
```

### 2. Detector Avanzado con Microsoft Presidio

```python
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
import spacy

class PresidioPIIDetector:
    """Detector de PII usando Microsoft Presidio"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        
        # Configurar NLP engine
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }
        
        # Inicializar engines
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()
        
        # Entidades a detectar
        self.default_entities = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "CRYPTO", "DATE_TIME", "IBAN_CODE", "IP_ADDRESS", 
            "LOCATION", "MEDICAL_LICENSE", "SSN", "UK_NHS",
            "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN",
            "US_PASSPORT", "URL", "BITCOIN_ADDRESS"
        ]
    
    def detect(self, text: str, entities: List[str] = None) -> List[Dict[str, Any]]:
        """Detectar PII usando Presidio"""
        
        entities_to_analyze = entities or self.default_entities
        
        # Analizar texto
        results = self.analyzer.analyze(
            text=text,
            entities=entities_to_analyze,
            language=self.language
        )
        
        # Convertir resultados a formato estándar
        pii_matches = []
        for result in results:
            pii_matches.append({
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'score': result.score,
                'text': text[result.start:result.end],
                'recognition_metadata': result.recognition_metadata
            })
        
        return pii_matches
    
    def anonymize(self, text: str, entities: List[str] = None) -> Dict[str, Any]:
        """Anonimizar PII en el texto"""
        
        # Detectar PII
        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=entities or self.default_entities,
            language=self.language
        )
        
        # Anonimizar
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        )
        
        return {
            'anonymized_text': anonymized_result.text,
            'entities_found': len(analyzer_results),
            'items': [
                {
                    'entity_type': item.entity_type,
                    'start': item.start,
                    'end': item.end,
                    'anonymized_text': item.text
                }
                for item in anonymized_result.items
            ]
        }
    
    def custom_anonymize(self, text: str, anonymization_config: Dict[str, str]) -> str:
        """Anonimización personalizada"""
        
        analyzer_results = self.analyzer.analyze(text=text, language=self.language)
        
        # Configurar operadores de anonimización
        operators = {}
        for entity_type, operation in anonymization_config.items():
            if operation == "replace":
                operators[entity_type] = {"type": "replace", "new_value": f"<{entity_type}>"}
            elif operation == "redact":
                operators[entity_type] = {"type": "redact"}
            elif operation == "hash":
                operators[entity_type] = {"type": "hash"}
            elif operation == "mask":
                operators[entity_type] = {"type": "mask", "masking_char": "*", "chars_to_mask": 4}
        
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
            operators=operators
        )
        
        return anonymized_result.text
```

### 3. Detector Híbrido Completo

```python
class ComprehensivePIIDetector:
    """Detector completo que combina múltiples métodos"""
    
    def __init__(self):
        self.regex_detector = RegexPIIDetector()
        self.presidio_detector = PresidioPIIDetector()
        
        # Configuración de confianza mínima por tipo
        self.confidence_thresholds = {
            PIIType.EMAIL: 0.8,
            PIIType.PHONE: 0.7,
            PIIType.CREDIT_CARD: 0.9,
            PIIType.SSN: 0.85,
            PIIType.IBAN: 0.85,
            'PERSON': 0.7,
            'LOCATION': 0.6
        }
    
    def detect_all_pii(self, text: str) -> Dict[str, Any]:
        """Detectar PII usando todos los métodos disponibles"""
        
        # Detección con regex
        regex_matches = self.regex_detector.detect(text)
        
        # Detección con Presidio
        presidio_matches = self.presidio_detector.detect(text)
        
        # Combinar y dedupe resultados
        combined_results = self._combine_results(regex_matches, presidio_matches)
        
        # Filtrar por confianza
        filtered_results = self._filter_by_confidence(combined_results)
        
        # Análisis de riesgo
        risk_assessment = self._assess_privacy_risk(filtered_results)
        
        return {
            'pii_detected': len(filtered_results) > 0,
            'total_entities': len(filtered_results),
            'entities': filtered_results,
            'risk_assessment': risk_assessment,
            'recommendations': self._generate_recommendations(filtered_results)
        }
    
    def _combine_results(self, regex_matches: List[PIIMatch], presidio_matches: List[Dict]) -> List[Dict]:
        """Combinar resultados de diferentes detectores"""
        
        combined = []
        
        # Convertir regex matches
        for match in regex_matches:
            combined.append({
                'entity_type': match.pii_type.value,
                'text': match.value,
                'start': match.start,
                'end': match.end,
                'confidence': match.confidence,
                'detector': 'regex',
                'context': match.context
            })
        
        # Añadir presidio matches
        for match in presidio_matches:
            combined.append({
                'entity_type': match['entity_type'].lower(),
                'text': match['text'],
                'start': match['start'],
                'end': match['end'],
                'confidence': match['score'],
                'detector': 'presidio',
                'context': None
            })
        
        # Dedupe por posición
        return self._dedupe_by_position(combined)
    
    def _dedupe_by_position(self, matches: List[Dict]) -> List[Dict]:
        """Eliminar duplicados basándose en posición"""
        
        if not matches:
            return matches
        
        matches.sort(key=lambda x: (x['start'], x['end']))
        
        deduplicated = []
        for match in matches:
            # Verificar overlap con matches previos
            overlaps = [
                m for m in deduplicated 
                if not (match['end'] <= m['start'] or match['start'] >= m['end'])
            ]
            
            if not overlaps:
                deduplicated.append(match)
            else:
                # Mantener el de mayor confianza
                best_match = max(overlaps + [match], key=lambda x: x['confidence'])
                
                # Remover overlaps anteriores y añadir el mejor
                deduplicated = [m for m in deduplicated if m not in overlaps]
                deduplicated.append(best_match)
        
        return sorted(deduplicated, key=lambda x: x['start'])
    
    def _filter_by_confidence(self, matches: List[Dict]) -> List[Dict]:
        """Filtrar matches por confianza mínima"""
        
        filtered = []
        for match in matches:
            entity_type = match['entity_type']
            threshold = self.confidence_thresholds.get(entity_type, 0.7)
            
            if match['confidence'] >= threshold:
                filtered.append(match)
        
        return filtered
    
    def _assess_privacy_risk(self, matches: List[Dict]) -> Dict[str, Any]:
        """Evaluar riesgo de privacidad"""
        
        if not matches:
            return {'level': 'NONE', 'score': 0.0, 'factors': []}
        
        risk_scores = {
            'email': 0.6,
            'phone': 0.6,
            'credit_card': 1.0,
            'ssn': 1.0,
            'iban': 0.9,
            'person': 0.5,
            'location': 0.4,
            'ip_address': 0.3
        }
        
        total_risk = 0.0
        risk_factors = []
        
        entity_counts = {}
        for match in matches:
            entity_type = match['entity_type']
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            # Calcular riesgo
            base_risk = risk_scores.get(entity_type, 0.5)
            confidence_factor = match['confidence']
            entity_risk = base_risk * confidence_factor
            
            total_risk += entity_risk
            risk_factors.append(f"{entity_type}: {entity_risk:.2f}")
        
        # Penalizar múltiples tipos de PII
        if len(entity_counts) > 3:
            total_risk *= 1.5
            risk_factors.append("Multiple PII types detected")
        
        # Determinar nivel de riesgo
        if total_risk >= 2.0:
            level = 'CRITICAL'
        elif total_risk >= 1.0:
            level = 'HIGH'
        elif total_risk >= 0.5:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {
            'level': level,
            'score': min(total_risk, 3.0),  # Cap at 3.0
            'factors': risk_factors,
            'entity_counts': entity_counts
        }
    
    def _generate_recommendations(self, matches: List[Dict]) -> List[str]:
        """Generar recomendaciones basadas en PII detectado"""
        
        if not matches:
            return ["No PII detected. Content appears safe for processing."]
        
        recommendations = []
        entity_types = set(match['entity_type'] for match in matches)
        
        if 'credit_card' in entity_types or 'ssn' in entity_types:
            recommendations.append("CRITICAL: Remove or encrypt financial/identity information before processing")
        
        if 'email' in entity_types or 'phone' in entity_types:
            recommendations.append("Consider anonymizing contact information")
        
        if 'person' in entity_types:
            recommendations.append("Implement consent mechanisms for personal names")
        
        if len(matches) > 5:
            recommendations.append("High volume of PII detected - consider full anonymization pipeline")
        
        recommendations.append("Ensure compliance with applicable privacy regulations (GDPR, CCPA)")
        recommendations.append("Implement data retention and deletion policies")
        
        return recommendations
```

## 📋 GDPR Compliance Framework

### 1. Configuración de Compliance

```python
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

class ConsentType(Enum):
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    LEGITIMATE_INTEREST = "legitimate_interest"

class DataPurpose(Enum):
    AI_TRAINING = "ai_training"
    SERVICE_IMPROVEMENT = "service_improvement"
    ANALYTICS = "analytics"
    MARKETING = "marketing"

@dataclass
class ConsentRecord:
    user_id: str
    purposes: List[DataPurpose]
    consent_type: ConsentType
    timestamp: datetime
    ip_address: str
    user_agent: str
    expiry_date: Optional[datetime] = None
    withdrawn: bool = False

class GDPRComplianceManager:
    """Gestor de compliance con GDPR"""
    
    def __init__(self):
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_retention_policies = {
            DataPurpose.AI_TRAINING: timedelta(days=730),  # 2 años
            DataPurpose.SERVICE_IMPROVEMENT: timedelta(days=365),  # 1 año
            DataPurpose.ANALYTICS: timedelta(days=90),  # 3 meses
            DataPurpose.MARKETING: timedelta(days=1095)  # 3 años
        }
    
    def record_consent(self, 
                      user_id: str, 
                      purposes: List[DataPurpose],
                      consent_type: ConsentType = ConsentType.EXPLICIT,
                      ip_address: str = "",
                      user_agent: str = "") -> bool:
        """Registrar consentimiento del usuario"""
        
        expiry_date = None
        if consent_type == ConsentType.EXPLICIT:
            # Consentimiento explícito expira en 2 años
            expiry_date = datetime.now() + timedelta(days=730)
        
        self.consent_records[user_id] = ConsentRecord(
            user_id=user_id,
            purposes=purposes,
            consent_type=consent_type,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            expiry_date=expiry_date
        )
        
        return True
    
    def check_consent(self, user_id: str, purpose: DataPurpose) -> Dict[str, Any]:
        """Verificar si existe consentimiento válido"""
        
        if user_id not in self.consent_records:
            return {
                'has_consent': False,
                'reason': 'No consent record found'
            }
        
        record = self.consent_records[user_id]
        
        # Verificar si el consentimiento fue retirado
        if record.withdrawn:
            return {
                'has_consent': False,
                'reason': 'Consent withdrawn'
            }
        
        # Verificar expiración
        if record.expiry_date and datetime.now() > record.expiry_date:
            return {
                'has_consent': False,
                'reason': 'Consent expired',
                'expired_on': record.expiry_date
            }
        
        # Verificar propósito específico
        if purpose not in record.purposes:
            return {
                'has_consent': False,
                'reason': f'No consent for purpose: {purpose.value}'
            }
        
        return {
            'has_consent': True,
            'consent_type': record.consent_type.value,
            'granted_on': record.timestamp,
            'expires_on': record.expiry_date
        }
    
    def withdraw_consent(self, user_id: str) -> bool:
        """Retirar consentimiento (Right to Withdraw)"""
        
        if user_id in self.consent_records:
            self.consent_records[user_id].withdrawn = True
            return True
        
        return False
    
    def delete_user_data(self, user_id: str) -> Dict[str, Any]:
        """Eliminar datos del usuario (Right to Erasure)"""
        
        result = {
            'user_id': user_id,
            'deleted_from': [],
            'errors': []
        }
        
        try:
            # Eliminar registros de consentimiento
            if user_id in self.consent_records:
                del self.consent_records[user_id]
                result['deleted_from'].append('consent_records')
            
            # Aquí se eliminarían datos de otros sistemas
            # - Base de datos de conversaciones
            # - Logs de sistema
            # - Cache de embeddings
            # - Modelos personalizados
            
            result['status'] = 'completed'
            result['timestamp'] = datetime.now()
            
        except Exception as e:
            result['errors'].append(str(e))
            result['status'] = 'failed'
        
        return result
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Exportar datos del usuario (Right to Portability)"""
        
        user_data = {
            'user_id': user_id,
            'export_timestamp': datetime.now(),
            'data_sources': {}
        }
        
        # Exportar registro de consentimiento
        if user_id in self.consent_records:
            record = self.consent_records[user_id]
            user_data['data_sources']['consent'] = {
                'purposes': [p.value for p in record.purposes],
                'consent_type': record.consent_type.value,
                'granted_on': record.timestamp,
                'ip_address': record.ip_address,
                'withdrawn': record.withdrawn
            }
        
        # Aquí se exportarían datos de otros sistemas
        # user_data['data_sources']['conversations'] = [...]
        # user_data['data_sources']['preferences'] = {...}
        
        return user_data
    
    def check_data_retention(self) -> List[Dict[str, Any]]:
        """Verificar políticas de retención de datos"""
        
        expired_data = []
        
        for user_id, record in self.consent_records.items():
            for purpose in record.purposes:
                retention_period = self.data_retention_policies.get(purpose)
                if retention_period:
                    expiry_date = record.timestamp + retention_period
                    
                    if datetime.now() > expiry_date:
                        expired_data.append({
                            'user_id': user_id,
                            'purpose': purpose.value,
                            'granted_on': record.timestamp,
                            'expired_on': expiry_date,
                            'days_overdue': (datetime.now() - expiry_date).days
                        })
        
        return expired_data
```

### 2. Pipeline de Anonimización

```python
class DataAnonymizationPipeline:
    """Pipeline completo de anonimización de datos"""
    
    def __init__(self):
        self.pii_detector = ComprehensivePIIDetector()
        self.gdpr_manager = GDPRComplianceManager()
    
    def process_text(self, 
                    text: str, 
                    user_id: str = None,
                    purpose: DataPurpose = DataPurpose.AI_TRAINING,
                    anonymization_level: str = "standard") -> Dict[str, Any]:
        """Procesar texto con anonimización completa"""
        
        # 1. Verificar consentimiento si hay user_id
        consent_check = None
        if user_id:
            consent_check = self.gdpr_manager.check_consent(user_id, purpose)
            
            if not consent_check['has_consent']:
                return {
                    'processed': False,
                    'reason': 'No valid consent',
                    'consent_status': consent_check
                }
        
        # 2. Detectar PII
        pii_analysis = self.pii_detector.detect_all_pii(text)
        
        # 3. Determinar nivel de anonimización necesario
        anonymization_strategy = self._determine_anonymization_strategy(
            pii_analysis, anonymization_level
        )
        
        # 4. Aplicar anonimización
        anonymized_text = self._apply_anonymization(text, pii_analysis, anonymization_strategy)
        
        # 5. Verificación post-anonimización
        verification = self._verify_anonymization(anonymized_text)
        
        return {
            'processed': True,
            'original_text': text,
            'anonymized_text': anonymized_text,
            'pii_analysis': pii_analysis,
            'anonymization_strategy': anonymization_strategy,
            'verification': verification,
            'consent_status': consent_check,
            'compliance_notes': self._generate_compliance_notes(pii_analysis, purpose)
        }
    
    def _determine_anonymization_strategy(self, 
                                        pii_analysis: Dict[str, Any], 
                                        level: str) -> Dict[str, str]:
        """Determinar estrategia de anonimización"""
        
        strategies = {
            'minimal': {
                'credit_card': 'mask',
                'ssn': 'mask', 
                'email': 'redact',
                'phone': 'redact'
            },
            'standard': {
                'credit_card': 'redact',
                'ssn': 'redact',
                'email': 'hash',
                'phone': 'hash',
                'person': 'replace',
                'location': 'generalize'
            },
            'aggressive': {
                'credit_card': 'redact',
                'ssn': 'redact',
                'email': 'redact',
                'phone': 'redact',
                'person': 'redact',
                'location': 'redact',
                'ip_address': 'redact'
            }
        }
        
        base_strategy = strategies.get(level, strategies['standard'])
        
        # Ajustar basado en el riesgo detectado
        risk_level = pii_analysis['risk_assessment']['level']
        
        if risk_level == 'CRITICAL':
            # Forzar redacción para entidades críticas
            for entity in ['credit_card', 'ssn', 'iban']:
                if entity in base_strategy:
                    base_strategy[entity] = 'redact'
        
        return base_strategy
    
    def _apply_anonymization(self, 
                           text: str, 
                           pii_analysis: Dict[str, Any], 
                           strategy: Dict[str, str]) -> str:
        """Aplicar anonimización según la estrategia"""
        
        anonymized_text = text
        
        # Ordenar entidades por posición (de atrás hacia adelante para mantener índices)
        entities = sorted(pii_analysis['entities'], key=lambda x: x['start'], reverse=True)
        
        for entity in entities:
            entity_type = entity['entity_type']
            start, end = entity['start'], entity['end']
            original_value = entity['text']
            
            # Obtener operación de anonimización
            operation = strategy.get(entity_type, 'replace')
            
            # Aplicar operación
            if operation == 'redact':
                replacement = '[REDACTED]'
            elif operation == 'mask':
                replacement = self._mask_value(original_value, entity_type)
            elif operation == 'hash':
                replacement = f"[HASH:{hash(original_value) % 10000:04d}]"
            elif operation == 'replace':
                replacement = f"[{entity_type.upper()}]"
            elif operation == 'generalize':
                replacement = self._generalize_value(original_value, entity_type)
            else:
                replacement = original_value  # No cambio
            
            # Reemplazar en el texto
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
        
        return anonymized_text
    
    def _mask_value(self, value: str, entity_type: str) -> str:
        """Enmascarar valor preservando formato"""
        
        if entity_type == 'credit_card':
            # Mostrar solo últimos 4 dígitos
            return f"****-****-****-{value[-4:]}"
        elif entity_type == 'phone':
            # Mostrar solo últimos 3 dígitos
            return f"***-***-{value[-3:]}"
        elif entity_type == 'email':
            # Mostrar solo dominio
            if '@' in value:
                return f"***@{value.split('@')[1]}"
        
        # Enmascaramiento genérico
        if len(value) <= 4:
            return '*' * len(value)
        else:
            return value[:2] + '*' * (len(value) - 4) + value[-2:]
    
    def _generalize_value(self, value: str, entity_type: str) -> str:
        """Generalizar valor para reducir especificidad"""
        
        if entity_type == 'location':
            # Generalizar ubicaciones a nivel de ciudad/región
            return '[CITY]'
        elif entity_type == 'date':
            # Generalizar fechas a año
            return '[YEAR]'
        
        return f"[{entity_type.upper()}]"
    
    def _verify_anonymization(self, anonymized_text: str) -> Dict[str, Any]:
        """Verificar que la anonimización fue efectiva"""
        
        # Re-detectar PII en texto anonimizado
        remaining_pii = self.pii_detector.detect_all_pii(anonymized_text)
        
        return {
            'pii_remaining': remaining_pii['pii_detected'],
            'entities_remaining': remaining_pii['total_entities'],
            'risk_level': remaining_pii['risk_assessment']['level'],
            'verification_passed': not remaining_pii['pii_detected']
        }
    
    def _generate_compliance_notes(self, 
                                 pii_analysis: Dict[str, Any], 
                                 purpose: DataPurpose) -> List[str]:
        """Generar notas de compliance"""
        
        notes = []
        
        if pii_analysis['pii_detected']:
            notes.append(f"PII detected for purpose: {purpose.value}")
            notes.append("Anonymization applied per GDPR requirements")
            
            risk_level = pii_analysis['risk_assessment']['level']
            if risk_level in ['HIGH', 'CRITICAL']:
                notes.append("High-risk PII detected - enhanced anonymization applied")
        
        notes.append("Data processed in compliance with applicable privacy regulations")
        notes.append("User rights (access, portability, erasure) are available upon request")
        
        return notes
```

## ✅ Checklist de Compliance

### GDPR Compliance:
- [ ] **Consentimiento explícito** implementado
- [ ] **Detectores de PII** funcionando (>90% accuracy)
- [ ] **Anonimización automática** configurada
- [ ] **Right to Erasure** implementado
- [ ] **Right to Portability** implementado
- [ ] **Data retention policies** configuradas
- [ ] **Breach notification** procedures establecidos

### Técnico:
- [ ] **Pipeline de anonimización** end-to-end
- [ ] **Verificación post-anonimización** automática
- [ ] **Audit logs** de todas las operaciones
- [ ] **Métricas de compliance** monitoreadas
- [ ] **Testing automatizado** de detección PII

## 🚀 Próximo Paso

En el **Laboratorio 4** implementaremos un sistema completo de detección de PII con Microsoft Presidio y crearemos un pipeline de compliance automatizado.

## 📖 Recursos Adicionales

- [GDPR Official Text](https://gdpr.eu/tag/gdpr/)
- [Microsoft Presidio Documentation](https://microsoft.github.io/presidio/)
- [CCPA Compliance Guide](https://oag.ca.gov/privacy/ccpa)
- [Privacy by Design Principles](https://www.ipc.on.ca/wp-content/uploads/resources/7foundationalprinciples.pdf)
