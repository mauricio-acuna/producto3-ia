# Security Configuration Template

## Safety Configuration (safety.yaml)

```yaml
# LLM Safety Configuration
safety:
  enabled: true
  
  # Prompt Injection Detection
  prompt_injection:
    enabled: true
    threshold: 0.8
    models:
      - "prompt-injection-classifier"
    block_on_detection: true
    
  # Content Filtering
  content_filter:
    enabled: true
    categories:
      - hate_speech
      - violence
      - sexual_content
      - harmful_advice
    
  # PII Detection
  pii_detection:
    enabled: true
    redaction: true
    types:
      - email
      - phone
      - ssn
      - credit_card
      - ip_address
      - person_name
    
  # Rate Limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    requests_per_hour: 1000
    burst_size: 10

# Input Sanitization
sanitization:
  enabled: true
  max_length: 4000
  allowed_formats:
    - text/plain
    - application/json
  
  # Character filtering
  filters:
    - remove_html_tags
    - escape_special_chars
    - normalize_unicode
    
# Output Validation
output_validation:
  enabled: true
  max_response_length: 8000
  content_validation: true
  pii_check: true
```

## Environment Variables for Security

```bash
# API Keys (use secrets management in production)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Security Settings
ENABLE_SAFETY_FILTERS=true
PII_DETECTION_ENDPOINT=https://your-pii-service.com
PROMPT_INJECTION_MODEL=microsoft/DialoGPT-medium

# Logging (avoid logging sensitive data)
LOG_LEVEL=INFO
LOG_PII=false
LOG_PROMPTS=false
```

## Python Security Implementation

```python
import re
from typing import List, Dict, Any
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class SecurityGuard:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
    def detect_prompt_injection(self, text: str) -> Dict[str, Any]:
        """Detect potential prompt injection attempts"""
        # Simple heuristic-based detection
        injection_patterns = [
            r"ignore previous instructions",
            r"disregard.*instructions",
            r"forget everything",
            r"new instructions:",
            r"system.*prompt",
            r"\\n\\n.*assistant",
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return {
                    "detected": True,
                    "confidence": 0.9,
                    "pattern": pattern
                }
        
        return {"detected": False, "confidence": 0.0}
    
    def detect_pii(self, text: str) -> List[Dict]:
        """Detect personally identifiable information"""
        results = self.analyzer.analyze(
            text=text,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
            language='en'
        )
        return [{"entity": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results]
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input"""
        # Remove potentially dangerous characters
        text = re.sub(r'[<>"\']', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length
        if len(text) > 4000:
            text = text[:4000]
            
        return text
    
    def anonymize_pii(self, text: str) -> str:
        """Anonymize detected PII"""
        analyzer_results = self.analyzer.analyze(text=text, language='en')
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        )
        return anonymized_result.text
```
