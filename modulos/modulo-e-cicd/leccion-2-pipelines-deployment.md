# 🚀 Lección 2: Pipelines CI/CD y Deployment Automation

## 🎯 Objetivos de la Lección

Al finalizar esta lección, serás capaz de:
- Diseñar pipelines CI/CD completos para sistemas LLM
- Implementar deployment automation con rollback
- Configurar entornos multi-stage (dev/staging/prod)
- Gestionar secretos y configuraciones
- Implementar blue-green y canary deployments
- Crear monitoring y alertas post-deployment

## 🔄 Pipeline Architecture

### 1. GitHub Actions Workflow Completo

```yaml
# .github/workflows/llm-cicd-pipeline.yml
name: LLM CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily regression tests

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  PYTHON_VERSION: "3.11"

jobs:
  # ========================================
  # STAGE 1: TESTING & QUALITY GATES
  # ========================================
  quality-gates:
    runs-on: ubuntu-latest
    outputs:
      pass-rate: ${{ steps.evaluation.outputs.pass_rate }}
      gate-status: ${{ steps.evaluation.outputs.overall_status }}
      
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        
    - name: Run unit tests
      run: |
        pytest tests/unit --cov=src --cov-report=xml --junitxml=test-results.xml
        
    - name: Run integration tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        TEST_DATABASE_URL: ${{ secrets.TEST_DATABASE_URL }}
      run: |
        pytest tests/integration -v --junitxml=integration-results.xml
        
    - name: Run LLM evaluations
      id: evaluation
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python scripts/run_quality_gates.py \
          --test-cases tests/data/evaluation_cases.json \
          --config configs/quality_gates.yaml \
          --model gpt-3.5-turbo \
          --output reports/quality_report.json \
          --fail-on-error
          
    - name: Upload evaluation reports
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-reports
        path: reports/
        
    - name: Publish test results
      if: always()
      uses: dorny/test-reporter@v1
      with:
        name: Test Results
        path: '*-results.xml'
        reporter: java-junit

  # ========================================
  # STAGE 2: SECURITY & COMPLIANCE
  # ========================================
  security-scan:
    runs-on: ubuntu-latest
    needs: quality-gates
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
        
    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD

  # ========================================
  # STAGE 3: BUILD & CONTAINERIZATION
  # ========================================
  build-image:
    runs-on: ubuntu-latest
    needs: [quality-gates, security-scan]
    if: needs.quality-gates.outputs.gate-status == 'pass'
    
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      image-tag: ${{ steps.meta.outputs.tags }}
      
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
          
    - name: Build and push
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64

  # ========================================
  # STAGE 4: DEPLOY TO STAGING
  # ========================================
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-image
    environment: staging
    
    steps:
    - name: Checkout deployment configs
      uses: actions/checkout@v4
      with:
        path: deployment
        
    - name: Setup Kubernetes
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
        
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG_STAGING }}" | base64 -d > ~/.kube/config
        
    - name: Deploy to staging
      run: |
        # Update deployment with new image
        kubectl set image deployment/llm-service \
          llm-service=${{ needs.build-image.outputs.image-tag }} \
          -n staging
          
        # Wait for rollout
        kubectl rollout status deployment/llm-service -n staging --timeout=300s
        
    - name: Run smoke tests
      run: |
        python scripts/smoke_tests.py --environment staging
        
    - name: Run performance tests
      run: |
        python scripts/load_tests.py \
          --target staging \
          --duration 300 \
          --concurrent-users 10

  # ========================================
  # STAGE 5: PRODUCTION DEPLOYMENT
  # ========================================
  deploy-production:
    runs-on: ubuntu-latest
    needs: [deploy-staging, build-image]
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout deployment configs
      uses: actions/checkout@v4
      
    - name: Setup Kubernetes
      uses: azure/setup-kubectl@v3
      
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBECONFIG_PRODUCTION }}" | base64 -d > ~/.kube/config
        
    - name: Blue-Green Deployment
      run: |
        python scripts/blue_green_deploy.py \
          --image ${{ needs.build-image.outputs.image-tag }} \
          --namespace production \
          --health-check-timeout 300
          
    - name: Update production traffic
      run: |
        # Gradually shift traffic to new version
        python scripts/traffic_manager.py \
          --shift-traffic \
          --percentage 10 \
          --monitor-duration 300
          
        python scripts/traffic_manager.py \
          --shift-traffic \
          --percentage 50 \
          --monitor-duration 300
          
        python scripts/traffic_manager.py \
          --shift-traffic \
          --percentage 100 \
          --monitor-duration 300

  # ========================================
  # STAGE 6: POST-DEPLOYMENT MONITORING
  # ========================================
  post-deployment-checks:
    runs-on: ubuntu-latest
    needs: deploy-production
    if: always()
    
    steps:
    - name: Wait for metrics stabilization
      run: sleep 300
      
    - name: Run post-deployment evaluations
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python scripts/post_deployment_eval.py \
          --environment production \
          --duration 600 \
          --alert-on-degradation
          
    - name: Create deployment summary
      run: |
        python scripts/create_deployment_summary.py \
          --deployment-id ${{ github.run_id }} \
          --environment production \
          --commit-sha ${{ github.sha }}
```

### 2. Blue-Green Deployment Script

```python
#!/usr/bin/env python3
"""
Blue-Green deployment script for Kubernetes
"""

import asyncio
import subprocess
import json
import time
import argparse
from typing import Dict, Any, Optional
import logging

class BlueGreenDeployer:
    """Implementación de Blue-Green deployment para Kubernetes"""
    
    def __init__(self, namespace: str = "production"):
        self.namespace = namespace
        self.logger = logging.getLogger("deployment.blue_green")
        
        # Configuración de colores
        self.colors = ["blue", "green"]
        
    async def deploy(self, 
                    image_tag: str,
                    health_check_timeout: int = 300) -> Dict[str, Any]:
        """Ejecutar deployment blue-green"""
        
        deployment_start = time.time()
        
        try:
            # 1. Determinar color actual y nuevo
            current_color = await self._get_current_color()
            new_color = self._get_opposite_color(current_color)
            
            self.logger.info(f"Starting blue-green deployment: {current_color} -> {new_color}")
            
            # 2. Deploy nueva versión al color inactivo
            await self._deploy_new_version(new_color, image_tag)
            
            # 3. Esperar que la nueva versión esté lista
            await self._wait_for_deployment(new_color, health_check_timeout)
            
            # 4. Ejecutar health checks
            health_check_result = await self._run_health_checks(new_color)
            
            if not health_check_result['healthy']:
                raise Exception(f"Health checks failed: {health_check_result['error']}")
            
            # 5. Cambiar tráfico al nuevo color
            await self._switch_traffic(new_color)
            
            # 6. Verificar métricas post-switch
            await self._monitor_post_switch(new_color)
            
            # 7. Limpiar versión anterior (opcional, con delay)
            # await self._schedule_cleanup(current_color)
            
            deployment_duration = time.time() - deployment_start
            
            return {
                'success': True,
                'previous_color': current_color,
                'new_color': new_color,
                'deployment_duration_seconds': deployment_duration,
                'health_checks': health_check_result,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            
            # Rollback si algo falló
            try:
                await self._rollback_traffic(current_color)
            except Exception as rollback_error:
                self.logger.error(f"Rollback also failed: {rollback_error}")
            
            return {
                'success': False,
                'error': str(e),
                'deployment_duration_seconds': time.time() - deployment_start
            }
    
    async def _get_current_color(self) -> str:
        """Determinar el color actualmente en producción"""
        
        try:
            # Obtener el selector del servicio principal
            cmd = [
                'kubectl', 'get', 'service', 'llm-service',
                '-n', self.namespace,
                '-o', 'jsonpath={.spec.selector.color}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            current_color = result.stdout.strip()
            
            if current_color not in self.colors:
                # Si no hay color definido, empezar con blue
                current_color = "blue"
                
            return current_color
            
        except subprocess.CalledProcessError:
            # Primer deployment, empezar con blue
            return "blue"
    
    def _get_opposite_color(self, color: str) -> str:
        """Obtener el color opuesto"""
        return "green" if color == "blue" else "blue"
    
    async def _deploy_new_version(self, color: str, image_tag: str):
        """Deployar nueva versión en el color especificado"""
        
        deployment_name = f"llm-service-{color}"
        
        # Crear/actualizar deployment
        deployment_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: {self.namespace}
  labels:
    app: llm-service
    color: {color}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-service
      color: {color}
  template:
    metadata:
      labels:
        app: llm-service
        color: {color}
    spec:
      containers:
      - name: llm-service
        image: {image_tag}
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: COLOR
          value: "{color}"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        
        # Aplicar deployment
        process = subprocess.run(
            ['kubectl', 'apply', '-f', '-'],
            input=deployment_yaml,
            text=True,
            capture_output=True
        )
        
        if process.returncode != 0:
            raise Exception(f"Failed to deploy {deployment_name}: {process.stderr}")
        
        self.logger.info(f"Deployed {deployment_name} with image {image_tag}")
    
    async def _wait_for_deployment(self, color: str, timeout: int):
        """Esperar que el deployment esté listo"""
        
        deployment_name = f"llm-service-{color}"
        
        cmd = [
            'kubectl', 'rollout', 'status', f'deployment/{deployment_name}',
            '-n', self.namespace,
            f'--timeout={timeout}s'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Deployment {deployment_name} failed to become ready: {result.stderr}")
        
        self.logger.info(f"Deployment {deployment_name} is ready")
    
    async def _run_health_checks(self, color: str) -> Dict[str, Any]:
        """Ejecutar health checks en la nueva versión"""
        
        try:
            # Obtener un pod del deployment
            cmd = [
                'kubectl', 'get', 'pods',
                '-n', self.namespace,
                '-l', f'app=llm-service,color={color}',
                '-o', 'jsonpath={.items[0].metadata.name}'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pod_name = result.stdout.strip()
            
            if not pod_name:
                raise Exception(f"No pods found for color {color}")
            
            # Port forward para health check
            port_forward = subprocess.Popen([
                'kubectl', 'port-forward',
                f'pod/{pod_name}',
                '8080:8000',
                '-n', self.namespace
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Esperar que port forward esté listo
            await asyncio.sleep(5)
            
            try:
                # Ejecutar health checks
                import aiohttp
                
                async with aiohttp.ClientSession() as session:
                    # Health check
                    async with session.get('http://localhost:8080/health') as resp:
                        if resp.status != 200:
                            raise Exception(f"Health check failed: {resp.status}")
                        
                        health_data = await resp.json()
                    
                    # Readiness check
                    async with session.get('http://localhost:8080/ready') as resp:
                        if resp.status != 200:
                            raise Exception(f"Readiness check failed: {resp.status}")
                
                return {
                    'healthy': True,
                    'health_data': health_data,
                    'checks_passed': ['health', 'readiness']
                }
                
            finally:
                port_forward.terminate()
                port_forward.wait()
                
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _switch_traffic(self, new_color: str):
        """Cambiar tráfico al nuevo color"""
        
        # Actualizar el service selector
        cmd = [
            'kubectl', 'patch', 'service', 'llm-service',
            '-n', self.namespace,
            '-p', f'{{"spec":{{"selector":{{"color":"{new_color}"}}}}}}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to switch traffic to {new_color}: {result.stderr}")
        
        self.logger.info(f"Traffic switched to {new_color}")
    
    async def _monitor_post_switch(self, new_color: str):
        """Monitorear métricas después del switch"""
        
        # Esperar que las métricas se estabilicen
        await asyncio.sleep(30)
        
        # Aquí implementarías checks de métricas específicas
        # Por ejemplo, error rate, latencia, throughput
        
        self.logger.info(f"Post-switch monitoring completed for {new_color}")
    
    async def _rollback_traffic(self, previous_color: str):
        """Rollback del tráfico al color anterior"""
        
        self.logger.warning(f"Rolling back traffic to {previous_color}")
        await self._switch_traffic(previous_color)

async def main():
    parser = argparse.ArgumentParser(description="Blue-Green Deployment")
    parser.add_argument("--image", required=True, help="Container image tag")
    parser.add_argument("--namespace", default="production", help="Kubernetes namespace")
    parser.add_argument("--health-check-timeout", type=int, default=300, help="Health check timeout")
    
    args = parser.parse_args()
    
    deployer = BlueGreenDeployer(args.namespace)
    result = await deployer.deploy(args.image, args.health_check_timeout)
    
    if result['success']:
        print(f"✅ Deployment successful: {result}")
        exit(0)
    else:
        print(f"❌ Deployment failed: {result}")
        exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

### 3. Traffic Manager para Canary Deployments

```python
#!/usr/bin/env python3
"""
Traffic manager para deployments canary
"""

import asyncio
import subprocess
import json
import time
import argparse
from typing import Dict, Any, Optional
import logging

class CanaryTrafficManager:
    """Gestor de tráfico para canary deployments"""
    
    def __init__(self, namespace: str = "production"):
        self.namespace = namespace
        self.logger = logging.getLogger("traffic.manager")
    
    async def shift_traffic(self, 
                          target_version: str,
                          percentage: int,
                          monitor_duration: int = 300) -> Dict[str, Any]:
        """Cambiar porcentaje de tráfico a nueva versión"""
        
        try:
            # 1. Actualizar configuración de tráfico
            await self._update_traffic_split(target_version, percentage)
            
            # 2. Monitorear métricas durante el periodo especificado
            metrics = await self._monitor_metrics(monitor_duration)
            
            # 3. Evaluar si el cambio fue exitoso
            success = self._evaluate_metrics(metrics)
            
            if not success:
                # Rollback automático si las métricas están mal
                await self._rollback_traffic()
                
                return {
                    'success': False,
                    'percentage': percentage,
                    'reason': 'Metrics degradation detected',
                    'metrics': metrics
                }
            
            return {
                'success': True,
                'percentage': percentage,
                'target_version': target_version,
                'metrics': metrics,
                'duration': monitor_duration
            }
            
        except Exception as e:
            self.logger.error(f"Traffic shift failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _update_traffic_split(self, target_version: str, percentage: int):
        """Actualizar split de tráfico usando Istio"""
        
        # Configuración de Istio VirtualService
        virtual_service_yaml = f"""
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-service
  namespace: {self.namespace}
spec:
  hosts:
  - llm-service
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: llm-service
        subset: {target_version}
      weight: 100
  - route:
    - destination:
        host: llm-service
        subset: stable
      weight: {100 - percentage}
    - destination:
        host: llm-service
        subset: {target_version}
      weight: {percentage}
"""
        
        # Aplicar configuración
        process = subprocess.run(
            ['kubectl', 'apply', '-f', '-'],
            input=virtual_service_yaml,
            text=True,
            capture_output=True
        )
        
        if process.returncode != 0:
            raise Exception(f"Failed to update traffic split: {process.stderr}")
        
        self.logger.info(f"Updated traffic split: {percentage}% to {target_version}")
    
    async def _monitor_metrics(self, duration: int) -> Dict[str, Any]:
        """Monitorear métricas durante el deployment"""
        
        start_time = time.time()
        metrics = {
            'error_rate': [],
            'latency_p95': [],
            'throughput': [],
            'cpu_usage': [],
            'memory_usage': []
        }
        
        while time.time() - start_time < duration:
            try:
                # Obtener métricas de Prometheus
                current_metrics = await self._fetch_prometheus_metrics()
                
                for metric_name, value in current_metrics.items():
                    if metric_name in metrics:
                        metrics[metric_name].append({
                            'timestamp': time.time(),
                            'value': value
                        })
                
                await asyncio.sleep(30)  # Muestrear cada 30 segundos
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch metrics: {e}")
        
        return metrics
    
    async def _fetch_prometheus_metrics(self) -> Dict[str, float]:
        """Obtener métricas de Prometheus"""
        
        # Implementación simplificada - en producción usarías la API de Prometheus
        metrics = {
            'error_rate': 0.01,  # 1% error rate
            'latency_p95': 2.5,  # 2.5s P95 latency
            'throughput': 100.0, # 100 req/s
            'cpu_usage': 0.65,   # 65% CPU
            'memory_usage': 0.75 # 75% memory
        }
        
        return metrics
    
    def _evaluate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Evaluar si las métricas están dentro de los SLAs"""
        
        # Thresholds de SLA
        sla_thresholds = {
            'error_rate': 0.05,    # Max 5% error rate
            'latency_p95': 5.0,    # Max 5s P95 latency
            'cpu_usage': 0.85,     # Max 85% CPU
            'memory_usage': 0.90   # Max 90% memory
        }
        
        for metric_name, threshold in sla_thresholds.items():
            if metric_name in metrics:
                recent_values = [point['value'] for point in metrics[metric_name][-5:]]  # Últimos 5 puntos
                
                if recent_values:
                    avg_value = sum(recent_values) / len(recent_values)
                    
                    if metric_name == 'error_rate' and avg_value > threshold:
                        self.logger.warning(f"SLA violation: {metric_name} = {avg_value} > {threshold}")
                        return False
                    elif metric_name in ['cpu_usage', 'memory_usage'] and avg_value > threshold:
                        self.logger.warning(f"Resource violation: {metric_name} = {avg_value} > {threshold}")
                        return False
                    elif metric_name == 'latency_p95' and avg_value > threshold:
                        self.logger.warning(f"Latency violation: {metric_name} = {avg_value} > {threshold}")
                        return False
        
        return True
    
    async def _rollback_traffic(self):
        """Rollback completo del tráfico"""
        
        self.logger.warning("Performing automatic rollback")
        await self._update_traffic_split("stable", 0)

class DeploymentAutomation:
    """Automatización completa de deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger("deployment.automation")
    
    async def progressive_deployment(self, 
                                   image_tag: str,
                                   namespace: str = "production") -> Dict[str, Any]:
        """Ejecutar deployment progresivo (canary)"""
        
        traffic_manager = CanaryTrafficManager(namespace)
        
        # Fases de deployment progresivo
        phases = [
            {'percentage': 5, 'duration': 300},   # 5% por 5 minutos
            {'percentage': 25, 'duration': 600},  # 25% por 10 minutos
            {'percentage': 50, 'duration': 600},  # 50% por 10 minutos
            {'percentage': 100, 'duration': 300}  # 100% por 5 minutos
        ]
        
        deployment_results = []
        
        for i, phase in enumerate(phases):
            self.logger.info(f"Phase {i+1}: Shifting {phase['percentage']}% traffic")
            
            result = await traffic_manager.shift_traffic(
                target_version="canary",
                percentage=phase['percentage'],
                monitor_duration=phase['duration']
            )
            
            deployment_results.append({
                'phase': i + 1,
                'percentage': phase['percentage'],
                'result': result
            })
            
            if not result['success']:
                self.logger.error(f"Phase {i+1} failed, stopping deployment")
                return {
                    'success': False,
                    'failed_phase': i + 1,
                    'results': deployment_results
                }
        
        return {
            'success': True,
            'total_phases': len(phases),
            'results': deployment_results
        }

async def main():
    parser = argparse.ArgumentParser(description="Traffic Manager")
    parser.add_argument("--shift-traffic", action="store_true", help="Shift traffic")
    parser.add_argument("--percentage", type=int, default=10, help="Traffic percentage")
    parser.add_argument("--monitor-duration", type=int, default=300, help="Monitor duration")
    parser.add_argument("--progressive", action="store_true", help="Progressive deployment")
    parser.add_argument("--image", help="Container image for progressive deployment")
    
    args = parser.parse_args()
    
    if args.progressive:
        if not args.image:
            print("--image required for progressive deployment")
            exit(1)
            
        automation = DeploymentAutomation()
        result = await automation.progressive_deployment(args.image)
        
        if result['success']:
            print(f"✅ Progressive deployment completed: {result}")
        else:
            print(f"❌ Progressive deployment failed: {result}")
            exit(1)
    
    elif args.shift_traffic:
        manager = CanaryTrafficManager()
        result = await manager.shift_traffic(
            "canary", 
            args.percentage,
            args.monitor_duration
        )
        
        if result['success']:
            print(f"✅ Traffic shift successful: {result}")
        else:
            print(f"❌ Traffic shift failed: {result}")
            exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

### 4. Post-Deployment Monitoring

```python
#!/usr/bin/env python3
"""
Post-deployment monitoring y alertas
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class AlertRule:
    name: str
    metric: str
    threshold: float
    operator: str  # >, <, >=, <=
    duration_minutes: int
    severity: str  # critical, warning, info

class PostDeploymentMonitor:
    """Monitor post-deployment con alertas automáticas"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.logger = logging.getLogger("post.deployment.monitor")
        
        # Reglas de alertas por defecto
        self.alert_rules = [
            AlertRule("High Error Rate", "error_rate", 0.05, ">=", 5, "critical"),
            AlertRule("High Latency", "latency_p95", 5.0, ">=", 5, "critical"),
            AlertRule("Low Throughput", "throughput", 50.0, "<=", 10, "warning"),
            AlertRule("High CPU Usage", "cpu_usage", 0.85, ">=", 10, "warning"),
            AlertRule("High Memory Usage", "memory_usage", 0.90, ">=", 5, "critical"),
            AlertRule("LLM Quality Degradation", "llm_quality_score", 0.8, "<=", 15, "critical")
        ]
    
    async def monitor_deployment(self, 
                               deployment_id: str,
                               duration_minutes: int = 60,
                               alert_on_degradation: bool = True) -> Dict[str, Any]:
        """Monitorear deployment por un periodo específico"""
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        alerts_triggered = []
        metrics_history = []
        
        self.logger.info(f"Starting post-deployment monitoring for {duration_minutes} minutes")
        
        while time.time() < end_time:
            try:
                # Obtener métricas actuales
                current_metrics = await self._collect_metrics()
                
                # Evaluar reglas de alertas
                triggered_alerts = self._evaluate_alert_rules(current_metrics)
                
                if triggered_alerts:
                    alerts_triggered.extend(triggered_alerts)
                    
                    if alert_on_degradation:
                        await self._send_alerts(triggered_alerts, deployment_id)
                
                # Guardar métricas históricas
                metrics_history.append({
                    'timestamp': time.time(),
                    'metrics': current_metrics
                })
                
                # Esperar antes del siguiente check
                await asyncio.sleep(60)  # Check cada minuto
                
            except Exception as e:
                self.logger.error(f"Error during monitoring: {e}")
        
        # Generar reporte final
        monitoring_report = self._generate_monitoring_report(
            deployment_id,
            alerts_triggered,
            metrics_history,
            duration_minutes
        )
        
        return monitoring_report
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Recopilar métricas del sistema"""
        
        # En producción, esto se conectaría a Prometheus, DataDog, etc.
        # Por ahora, simulamos métricas
        
        metrics = {
            'error_rate': 0.02,
            'latency_p95': 2.1,
            'latency_p99': 3.5,
            'throughput': 85.0,
            'cpu_usage': 0.72,
            'memory_usage': 0.68,
            'disk_usage': 0.45,
            'llm_quality_score': 0.85,
            'active_connections': 150,
            'queue_depth': 5
        }
        
        return metrics
    
    def _evaluate_alert_rules(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Evaluar reglas de alertas contra métricas actuales"""
        
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if rule.metric not in metrics:
                continue
            
            metric_value = metrics[rule.metric]
            threshold_exceeded = False
            
            if rule.operator == ">":
                threshold_exceeded = metric_value > rule.threshold
            elif rule.operator == ">=":
                threshold_exceeded = metric_value >= rule.threshold
            elif rule.operator == "<":
                threshold_exceeded = metric_value < rule.threshold
            elif rule.operator == "<=":
                threshold_exceeded = metric_value <= rule.threshold
            
            if threshold_exceeded:
                alert = {
                    'rule_name': rule.name,
                    'metric': rule.metric,
                    'current_value': metric_value,
                    'threshold': rule.threshold,
                    'operator': rule.operator,
                    'severity': rule.severity,
                    'timestamp': datetime.now().isoformat(),
                    'environment': self.environment
                }
                
                triggered_alerts.append(alert)
                self.logger.warning(f"Alert triggered: {rule.name} - {metric_value} {rule.operator} {rule.threshold}")
        
        return triggered_alerts
    
    async def _send_alerts(self, alerts: List[Dict[str, Any]], deployment_id: str):
        """Enviar alertas a sistemas de notificación"""
        
        for alert in alerts:
            # Slack notification
            await self._send_slack_alert(alert, deployment_id)
            
            # Email notification para alertas críticas
            if alert['severity'] == 'critical':
                await self._send_email_alert(alert, deployment_id)
            
            # PagerDuty para alertas críticas
            if alert['severity'] == 'critical':
                await self._send_pagerduty_alert(alert, deployment_id)
    
    async def _send_slack_alert(self, alert: Dict[str, Any], deployment_id: str):
        """Enviar alerta a Slack"""
        
        # Implementación simplificada
        severity_emoji = {
            'critical': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
        
        emoji = severity_emoji.get(alert['severity'], '🔔')
        
        message = (
            f"{emoji} **{alert['severity'].upper()}** Alert\n"
            f"**Rule:** {alert['rule_name']}\n"
            f"**Environment:** {alert['environment']}\n"
            f"**Deployment:** {deployment_id}\n"
            f"**Metric:** {alert['metric']} = {alert['current_value']}\n"
            f"**Threshold:** {alert['operator']} {alert['threshold']}\n"
            f"**Time:** {alert['timestamp']}"
        )
        
        self.logger.info(f"Slack alert: {message}")
    
    async def _send_email_alert(self, alert: Dict[str, Any], deployment_id: str):
        """Enviar alerta por email"""
        
        # Implementación simplificada
        self.logger.info(f"Email alert sent for {alert['rule_name']}")
    
    async def _send_pagerduty_alert(self, alert: Dict[str, Any], deployment_id: str):
        """Enviar alerta a PagerDuty"""
        
        # Implementación simplificada
        self.logger.info(f"PagerDuty alert sent for {alert['rule_name']}")
    
    def _generate_monitoring_report(self, 
                                  deployment_id: str,
                                  alerts: List[Dict[str, Any]],
                                  metrics_history: List[Dict[str, Any]],
                                  duration_minutes: int) -> Dict[str, Any]:
        """Generar reporte de monitoreo"""
        
        # Calcular estadísticas de métricas
        metric_stats = {}
        
        if metrics_history:
            all_metrics = {}
            for entry in metrics_history:
                for metric_name, value in entry['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
            
            for metric_name, values in all_metrics.items():
                metric_stats[metric_name] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'final': values[-1] if values else 0
                }
        
        # Clasificar alertas por severidad
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in alerts if a['severity'] == 'warning']
        
        # Determinar estado general del deployment
        if critical_alerts:
            deployment_health = "critical"
        elif warning_alerts:
            deployment_health = "warning"
        else:
            deployment_health = "healthy"
        
        return {
            'deployment_id': deployment_id,
            'environment': self.environment,
            'monitoring_duration_minutes': duration_minutes,
            'deployment_health': deployment_health,
            'summary': {
                'total_alerts': len(alerts),
                'critical_alerts': len(critical_alerts),
                'warning_alerts': len(warning_alerts),
                'metric_checks': len(metrics_history)
            },
            'alerts': alerts,
            'metric_statistics': metric_stats,
            'recommendations': self._generate_recommendations(deployment_health, alerts),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, 
                                health: str,
                                alerts: List[Dict[str, Any]]) -> List[str]:
        """Generar recomendaciones basadas en el monitoreo"""
        
        recommendations = []
        
        if health == "critical":
            recommendations.append("Consider immediate rollback due to critical alerts")
            recommendations.append("Investigate root cause before next deployment")
        
        if any(a['metric'] == 'error_rate' for a in alerts):
            recommendations.append("Review error logs and fix underlying issues")
        
        if any(a['metric'].startswith('latency') for a in alerts):
            recommendations.append("Optimize performance or scale resources")
        
        if any(a['metric'] in ['cpu_usage', 'memory_usage'] for a in alerts):
            recommendations.append("Consider increasing resource limits")
        
        if any(a['metric'] == 'llm_quality_score' for a in alerts):
            recommendations.append("Review LLM model quality and evaluation criteria")
        
        if not recommendations:
            recommendations.append("Deployment appears healthy, continue monitoring")
        
        return recommendations

async def main():
    parser = argparse.ArgumentParser(description="Post-Deployment Monitor")
    parser.add_argument("--environment", default="production", help="Environment")
    parser.add_argument("--duration", type=int, default=60, help="Monitor duration in minutes")
    parser.add_argument("--deployment-id", help="Deployment ID")
    parser.add_argument("--alert-on-degradation", action="store_true", help="Send alerts")
    
    args = parser.parse_args()
    
    monitor = PostDeploymentMonitor(args.environment)
    
    deployment_id = args.deployment_id or f"deploy-{int(time.time())}"
    
    result = await monitor.monitor_deployment(
        deployment_id,
        args.duration,
        args.alert_on_degradation
    )
    
    print(json.dumps(result, indent=2))
    
    # Exit con código de error si hay problemas críticos
    if result['deployment_health'] == 'critical':
        exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

## 🔧 Configuraciones de Entorno

### 1. Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-production
  labels:
    istio-injection: enabled

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-service-config
  namespace: llm-production
data:
  environment: "production"
  log_level: "INFO"
  max_concurrent_requests: "100"
  model_cache_size: "1000"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: llm-service-secrets
  namespace: llm-production
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  database-url: <base64-encoded-url>

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-service
  namespace: llm-production
spec:
  selector:
    app: llm-service
    color: blue  # Will be updated by deployment scripts
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
  type: ClusterIP

---
# k8s/istio-gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: llm-service-gateway
  namespace: llm-production
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - llm-api.example.com

---
# k8s/destination-rule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: llm-service
  namespace: llm-production
spec:
  host: llm-service
  subsets:
  - name: blue
    labels:
      color: blue
  - name: green
    labels:
      color: green
  - name: stable
    labels:
      color: blue
  - name: canary
    labels:
      color: green
```

### 2. Terraform Infrastructure

```hcl
# terraform/main.tf
terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}

# Istio installation
resource "helm_release" "istio_base" {
  name       = "istio-base"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "base"
  namespace  = "istio-system"
  
  create_namespace = true
}

resource "helm_release" "istiod" {
  name       = "istiod"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "istiod"
  namespace  = "istio-system"
  
  depends_on = [helm_release.istio_base]
}

# Prometheus for monitoring
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  
  create_namespace = true
  
  values = [
    file("${path.module}/prometheus-values.yaml")
  ]
}

# Grafana dashboards
resource "kubernetes_config_map" "grafana_dashboards" {
  metadata {
    name      = "llm-dashboards"
    namespace = "monitoring"
    labels = {
      grafana_dashboard = "1"
    }
  }
  
  data = {
    "llm-metrics.json" = file("${path.module}/dashboards/llm-metrics.json")
  }
}
```

## ✅ Mejores Prácticas de CI/CD

### 1. **Pipeline Design**
- Etapas bien definidas y separadas
- Parallel execution donde sea posible
- Fail-fast strategies
- Comprehensive testing en cada etapa

### 2. **Deployment Strategies**
- Blue-Green para zero-downtime
- Canary para validación gradual
- Feature flags para control fino
- Automated rollback en fallos

### 3. **Monitoring & Observability**
- Health checks comprensivos
- SLA monitoring en tiempo real
- Alertas proactivas
- Post-deployment validation

### 4. **Security & Compliance**
- Secret management seguro
- Vulnerability scanning
- Compliance checking
- Audit trails completos

## 🎯 Próximo Paso

En el **Laboratorio 8** (Capstone Final) integraremos todos estos conceptos en un proyecto completo de extremo a extremo.

## 📖 Recursos Adicionales

- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [GitOps with ArgoCD](https://argo-cd.readthedocs.io/en/stable/)
- [Monitoring with Prometheus](https://prometheus.io/docs/practices/rules/)
