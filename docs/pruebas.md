# Pruebas y Resultados del Modelo

Este documento expone los resultados de la evaluación del modelo **Decision Tree Classifier** y los casos de prueba manuales ejecutados para validar su lógica.

## Resultados del Entrenamiento

El dataset de 1000 registros fue dividido en:

- **Datos de Entrenamiento:** 80% (800 registros)
- **Datos de Prueba:** 20% (200 registros)

El modelo logró aprender con éxito las reglas de negocio implementadas en la generación de datos.

### Métricas de Evaluación

- **Accuracy (Precisión General):** `99.50%`
- **F1-Score (Promedio macro):** `0.99`

*Análisis:* La altísima precisión se debe a que el modelo (Árbol de Decisión) es perfecto para mapear reglas lógicas condicionales. El árbol determinó correctamente que las variables más críticas para tomar la decisión eran el `tiempo` (<= 47.5 min), los `transbordos` (<= 1) y el `tráfico`.

***

## Casos de Prueba Lógicos

Para validar que el modelo aprendió correctamente a comportarse como un sistema inteligente basado en reglas, se ingresaron manualmente datos simulando 3 escenarios clave:

### Caso 1: Ruta Ideal

- **Condiciones:** Bajo tráfico, tiempo corto (20 min), sin transbordos.
- **Resultado Esperado:** Ruta Seleccionada (SI)
- **Predicción del Modelo:** ✅ `Ruta SELECCIONADA (SI)`

### Caso 2: Ruta Penalizada por Tráfico

- **Condiciones:** Alto tráfico, tiempo medio/alto (50 min).
- **Resultado Esperado:** Ruta Descartada (NO)
- **Predicción del Modelo:** ✅ `Ruta DESCARTADA (NO)`

### Caso 3: Desempate por Transbordos

Se simularon dos rutas idénticas en tiempo y tráfico, pero con diferente cantidad de transbordos.

- **Sub-caso 3.1 (Menos transbordos):** 40 min, tráfico medio, **1 transbordo**.
  - **Resultado Esperado:** Ruta Seleccionada.
  - **Predicción del Modelo:** ✅ `Ruta SELECCIONADA (SI)`
- **Sub-caso 3.2 (Más transbordos):** 40 min, tráfico medio, **3 transbordos**.
  - **Resultado Esperado:** Ruta Descartada.
  - **Predicción del Modelo:** ✅ `Ruta DESCARTADA (NO)`

###  Conclusión

El sistema demostró ser capaz de emular exitosamente la toma de decisiones humanas basada en prioridades de transporte (tiempo, comodidad y flujo vehicular) utilizando aprendizaje supervisado.
