# Descripción de los Datos (Dataset)

Este documento detalla la estructura y el origen de los datos utilizados para entrenar el modelo del **Sistema Inteligente de Selección de Rutas**.

## Origen de los Datos
Los datos fueron generados de manera sintética (simulada) mediante un script en Python (`src/generar_dataset.py`). Se crearon un total de **1,000 registros** que representan diferentes escenarios de rutas de transporte entre distintos puntos de una ciudad teórica.

## Diccionario de Variables

| Variable        | Tipo       | Descripción                                                                                                   |
| --------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| **origen**      | Categórica | Punto de inicio de la ruta (ej. Centro, Norte, Aeropuerto).                                                   |
| **destino**     | Categórica | Punto final de la ruta. Siempre distinto al origen.                                                           |
| **distancia**   | Numérica   | Distancia total de la ruta expresada en kilómetros (km). Rango simulado: 2.0 a 35.0 km.                       |
| **tiempo**      | Numérica   | Tiempo estimado de viaje en minutos. Se calcula base a la distancia y se ve afectado por el nivel de tráfico. |
| **trafico**     | Categórica | Nivel de congestión vehicular en la ruta. Valores posibles: `bajo`, `medio`, `alto`.                          |
| **transbordos** | Numérica   | Número de veces que el usuario debe cambiar de vehículo/línea. Rango: 0 a 3.                                  |
| **mejor\_ruta** | Categórica | **Variable Objetivo (Target).** Etiqueta que define si la ruta es la óptima (`SI`) o no (`NO`).               |

## Supuestos y Lógica de Generación

Para que el modelo de Machine Learning tuviera un patrón lógico que aprender, los datos no se generaron de forma 100% aleatoria. Se aplicaron las siguientes reglas de negocio:

1. **Cálculo del Tiempo:** El tiempo base es proporcional a la distancia. Sin embargo, el tráfico afecta directamente este valor:
   - Tráfico `medio`: Incrementa el tiempo un 30%.
   - Tráfico `alto`: Incrementa el tiempo un 80%.
2. **Etiquetado de la "Mejor Ruta":** Una ruta se etiquetó automáticamente como `SI` (Mejor ruta) si cumplía con las siguientes condiciones:
   - El tráfico **NO** es alto.
   - El tiempo total es **menor o igual a 45 minutos**.
   - Los transbordos son **1 o menos**.
   *(Se añadió una ligera variación estadística para rutas con tráfico bajo, 0 transbordos y hasta 60 minutos, simulando que a veces la comodidad prioriza sobre el tiempo, añadiendo ruido realista para el entrenamiento).*

