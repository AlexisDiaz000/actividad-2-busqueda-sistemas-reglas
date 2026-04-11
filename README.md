# Sistema Inteligente de Selección de Rutas (ML + Reglas)

Este proyecto es una implementación en Python de un sistema inteligente que determina la mejor ruta de transporte entre dos puntos. Combina la generación de datos simulados mediante **reglas lógicas** y el entrenamiento de **dos modelos de Machine Learning supervisados (Decision Tree y Random Forest)** para replicar esa toma de decisiones.

## Estructura del Proyecto

```text
proyecto-rutas/
│
├── data/
│   └── dataset.csv                # Datos simulados generados (1000 registros)
│
├── src/
│   ├── generar_dataset.py         # Script que crea los datos usando reglas lógicas
│   └── modelo.py                  # Script que entrena el árbol de decisión y ejecuta pruebas
│
├── docs/
│   ├── descripcion_datos.md       # Diccionario de variables y reglas de creación
│   └── pruebas.md                 # Métricas de rendimiento y casos de prueba manuales
│
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## Cómo ejecutar el proyecto

1. **Crear y activar el entorno virtual (Recomendado):**
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generar el dataset:**
   ```bash
   python src/generar_dataset.py
   ```
   *Esto creará el archivo `data/dataset.csv`.*

4. **Entrenar el modelo y probarlo:**
   ```bash
   python src/modelo.py
   ```
   *Esto imprimirá las métricas del modelo (accuracy), las reglas lógicas que aprendió el árbol y los resultados de los casos de prueba manuales.*

## Lógica del Sistema y Análisis
El sistema penaliza rutas con alto tráfico y múltiples transbordos, dando prioridad a trayectos cortos en tiempo. El proyecto incluye:
- **Comparativa de Modelos:** Entrenamiento de Árbol de Decisión y Random Forest para comparar su precisión.
- **Análisis de Importancia:** En la carpeta `docs/` se encuentra `importancia_variables.png`, un gráfico que demuestra qué variables pesaron más en la decisión de la Inteligencia Artificial.
- **Árbol de Decisión:** En la carpeta `docs/` encontrarás el archivo `arbol_decision.png` que muestra visualmente las reglas aprendidas por el modelo para clasificar las rutas.
