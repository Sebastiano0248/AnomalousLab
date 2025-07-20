# Resumen de Optimización de Modelos de Trayectorias

## Introducción

Este documento resume el proceso de optimización y entrenamiento de modelos de regresión para predecir parámetros específicos de diferentes tipos de trayectorias anómalas. Los modelos están basados en arquitecturas CNN optimizadas específicamente para cada tipo de trayectoria y parámetro.

## Metodología de Optimización

### Proceso General
1. **Búsqueda de Hiperparámetros**: Se utilizó un proceso sistemático de búsqueda de hiperparámetros para cada combinación de tipo de trayectoria y parámetro.
2. **Arquitecturas Específicas**: Cada modelo utiliza una arquitectura CNN optimizada específicamente para el tipo de trayectoria y parámetro objetivo.
3. **Evaluación**: Los modelos se evaluaron usando Mean Squared Error (MSE) como métrica principal.

### Configuraciones Testadas
- **Learning Rates**: [0.001, 0.0005, 0.0001, 0.00005, 0.00015, 0.002]
- **Batch Sizes**: [16, 32, 64, 128]
- **Epochs**: [20, 25, 50, 100]
- **Weight Decay**: [1e-5, 1e-4, 1e-3, 5e-3]
- **Número de Partículas**: [15000, 20000]
- **Pasos por Trayectoria**: [300, 500]

## Resultados de Optimización

### Top 3 Mejores Modelos

#### 1. Fractional Brownian Motion - Parámetro Hurst
- **MSE**: 0.0004 (Mejor rendimiento global)
- **Configuración Óptima**:
  - Learning Rate: 5e-05
  - Batch Size: 64
  - Epochs: 25
  - Weight Decay: 0.0001
  - Partículas: 15000
  - Pasos: 500
- **Tiempo de Entrenamiento**: 183.8 segundos
- **Arquitectura**: FBMHurstArchitecture (Específica para FBM)

#### 2. Scaled Brownian Motion - Parámetro Beta
- **MSE**: 0.0005
- **Configuración Óptima**:
  - Learning Rate: 0.0001
  - Batch Size: 64
  - Epochs: 25
  - Weight Decay: 0.0001
  - Partículas: 15000
  - Pasos: 500
- **Tiempo de Entrenamiento**: 33.2 segundos
- **Arquitectura**: SBMBetaArchitecture (Específica para SBM)

#### 3. Annealed Transit Time Model - Parámetro Gamma Shape
- **MSE**: 0.0052
- **Configuración Óptima**:
  - Learning Rate: 0.001
  - Batch Size: 128
  - Epochs: 25
  - Weight Decay: 0.001
  - Partículas: 20000
  - Pasos: 500
- **Tiempo de Entrenamiento**: 80.0 segundos
- **Arquitectura**: ATTMGammaShapeArchitecture (Específica para ATTM)

### Resultados Completos

| Tipo de Trayectoria | Parámetro | MSE | Tiempo (s) | Dificultad |
|---------------------|-----------|-----|------------|------------|
| FractionalBrownianMotion | hurst | 0.0004 | 183.8 | Muy Fácil |
| ScaledBrownianMotion | beta | 0.0005 | 33.2 | Muy Fácil |
| AnnealedTransitTimeModel | gamma_shape | 0.0052 | 80.0 | Fácil |
| ContinuousTimeRandomWalk | alpha | 0.0207 | 58.1 | Moderada |
| BrownianMotion | sigma | 0.0247 | 56.5 | Moderada |
| LevyWalks | alpha | 0.0386 | 69.0 | Moderada |
| LevyWalks | speed | 0.0488 | 33.8 | Difícil |
| BrownianMotion | delta_t | 0.0609 | 51.6 | Difícil |
| LevyWalks | beta | 0.0807 | 42.8 | Muy Difícil |

## Análisis de Patrones

### Factores de Éxito
1. **Arquitecturas Específicas**: Los mejores resultados se obtuvieron con arquitecturas CNN diseñadas específicamente para cada tipo de trayectoria.
2. **Learning Rates Moderados**: Los mejores modelos utilizaron learning rates entre 5e-05 y 0.001.
3. **Batch Sizes Medianos**: Batch sizes de 64 fueron más efectivos que tamaños extremos.
4. **Regularización Moderada**: Weight decay entre 1e-4 y 1e-3 mostró mejores resultados.

### Dificultad de Predicción
- **Parámetros más fáciles**: Hurst (FBM), Beta (SBM)
- **Parámetros más difíciles**: Beta (Lévy Walks), Delta_t (Brownian Motion)

### Eficiencia Computacional
- Los modelos más rápidos: SBM-Beta (33.2s), LW-Speed (33.8s)
- Los modelos más lentos: FBM-Hurst (183.8s) - pero con el mejor MSE

## Estructura de Modelos Nueva

Los modelos entrenados utilizan la nueva estructura de AnomalousLib que incluye:
- **Metadata completa**: Información sobre arquitectura, parámetro objetivo, y configuración
- **Compatibilidad**: Uso de `save_model()` y `load_model()` de AnomalousLib
- **Arquitecturas específicas**: Cada modelo usa una arquitectura optimizada para su caso de uso

## Archivos Generados

### Modelos Finales (best_models/)
- `FractionalBrownianMotionHurstRegressor.pth`
- `ScaledBrownianMotionBetaRegressor.pth`  
- `LevyWalksAlphaRegressor.pth`

### Datos de Entrenamiento
- Resultados completos disponibles en archivos CSV con timestamp
- Configuraciones intermedias guardadas por seguridad
- Métricas de progreso detalladas

## Recomendaciones

1. **Para Producción**: Utilizar los top 3 modelos (FBM-Hurst, SBM-Beta, ATTM-GammaShape)
2. **Para Investigación**: Explorar mejoras en modelos con MSE > 0.05
3. **Optimización Futura**: Enfocar en LevyWalks-Beta que mostró mayor dificultad

## Fecha de Última Actualización
19 de Julio de 2025

---
*Generado automáticamente a partir de los resultados de optimización de parámetros de trayectorias anómalas*
