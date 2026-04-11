# Pruebas y Resultados del Modelo

Este documento expone los resultados de la evaluación de los modelos de aprendizaje automático y los casos de prueba manuales ejecutados para validar su lógica.

## Resultados del Entrenamiento y Comparativa

El dataset de 1000 registros fue dividido en:
- **Datos de Entrenamiento:** 80% (800 registros)
- **Datos de Prueba:** 20% (200 registros)

Se entrenaron dos modelos distintos para comparar su rendimiento al aprender las reglas de negocio implementadas en la generación de datos.

### Precisión (Accuracy) de los Modelos
- **Árbol de Decisión (Decision Tree):** `99.50%`
- **Bosque Aleatorio (Random Forest):** `99.50%`

*Análisis:* Ambos modelos alcanzan una precisión casi perfecta. Esto ocurre porque el dataset está regido por reglas condicionales deterministas (tráfico, tiempo, transbordos), y los algoritmos basados en árboles son matemáticamente ideales para mapear y replicar este tipo de lógicas estrictas.

## Importancia de Variables (Feature Importance)
Al analizar el modelo Random Forest para entender **cómo tomó las decisiones**, se generó el gráfico `importancia_variables.png` (disponible en esta misma carpeta).

El modelo determinó que la variable más crítica es el **tiempo**, seguido muy de cerca por los **transbordos**, y finalmente el **tráfico**. La distancia en kilómetros no afectó directamente la clasificación de la "mejor ruta" (ya que la regla de negocio original se basaba en el tiempo total), demostrando que el modelo aprendió a ignorar las variables irrelevantes.

***

## Casos de Prueba Lógicos

Para validar que el modelo aprendió correctamente a comportarse como un sistema inteligente basado en reglas, se ingresaron manualmente datos simulando 3 escenarios clave:

### Caso 1: Ruta Ideal
- **Condiciones:** Bajo tráfico, tiempo corto (20 min), sin transbordos.
- **Resultado Esperado:** Ruta Seleccionada (SI)
- **Predicción del Modelo:** `Ruta SELECCIONADA (SI)`

### Caso 2: Ruta Penalizada por Tráfico
- **Condiciones:** Alto tráfico, tiempo medio/alto (50 min).
- **Resultado Esperado:** Ruta Descartada (NO)
- **Predicción del Modelo:** `Ruta DESCARTADA (NO)`

### Caso 3: Desempate por Transbordos
Se simularon dos rutas idénticas en tiempo y tráfico, pero con diferente cantidad de transbordos.

- **Sub-caso 3.1 (Menos transbordos):** 40 min, tráfico medio, **1 transbordo**.
  - **Resultado Esperado:** Ruta Seleccionada.
  - **Predicción del Modelo:** `Ruta SELECCIONADA (SI)`

- **Sub-caso 3.2 (Más transbordos):** 40 min, tráfico medio, **3 transbordos**.
  - **Resultado Esperado:** Ruta Descartada.
  - **Predicción del Modelo:** `Ruta DESCARTADA (NO)`

### Conclusión

El sistema demostró ser capaz de emular exitosamente la toma de decisiones humanas basada en prioridades de transporte (tiempo, comodidad y flujo vehicular) utilizando aprendizaje supervisado.
