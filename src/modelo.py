import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

def cargar_y_preprocesar():
    # Cargar dataset
    ruta_dataset = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    try:
        df = pd.read_csv(ruta_dataset)
        print(f"Dataset cargado correctamente. {df.shape[0]} registros encontrados.")
    except FileNotFoundError:
        print("Error: No se encontró el dataset. Ejecuta generar_dataset.py primero.")
        return None, None, None, None
        
    # Inicializar codificadores para variables categóricas
    le_origen = LabelEncoder()
    le_destino = LabelEncoder()
    le_trafico = LabelEncoder()
    
    # Aplicar codificación
    df['origen_cod'] = le_origen.fit_transform(df['origen'])
    df['destino_cod'] = le_destino.fit_transform(df['destino'])
    
    # Para el tráfico, le damos un orden lógico (bajo=0, medio=1, alto=2)
    mapeo_trafico = {'bajo': 0, 'medio': 1, 'alto': 2}
    df['trafico_cod'] = df['trafico'].map(mapeo_trafico)
    
    # La variable objetivo (SI=1, NO=0)
    df['mejor_ruta_cod'] = df['mejor_ruta'].map({'SI': 1, 'NO': 0})
    
    # Seleccionar las características (Features = X) y el objetivo (Target = y)
    # Excluimos origen y destino por ahora porque la regla de negocio se basó en variables numéricas,
    # pero las incluimos para que el modelo tenga todas las variables.
    X = df[['distancia', 'tiempo', 'trafico_cod', 'transbordos']]
    y = df['mejor_ruta_cod']
    
    # Diccionario para recordar el mapeo y usarlo en las pruebas
    codificadores = {
        'trafico': mapeo_trafico
    }
    
    return X, y, df, codificadores

def entrenar_modelo():
    X, y, df, codificadores = cargar_y_preprocesar()
    
    if X is None:
        return None, None
        
    print("\n--- FASE DE ENTRENAMIENTO Y COMPARATIVA DE MODELOS ---")
    # Dividir en conjunto de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo 1: Decision Tree (Árbol de Decisión)
    inicio_dt = time.time()
    modelo_dt = DecisionTreeClassifier(random_state=42, max_depth=4)
    modelo_dt.fit(X_train, y_train)
    y_pred_dt = modelo_dt.predict(X_test)
    fin_dt = time.time()
    tiempo_dt = (fin_dt - inicio_dt) * 1000 # en milisegundos
    acc_dt = accuracy_score(y_test, y_pred_dt)
    
    # Modelo 2: Random Forest (Bosque Aleatorio)
    inicio_rf = time.time()
    modelo_rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=4)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    fin_rf = time.time()
    tiempo_rf = (fin_rf - inicio_rf) * 1000 # en milisegundos
    acc_rf = accuracy_score(y_test, y_pred_rf)
    
    # Evaluar los modelos
    print("\n--- RESULTADOS DE PRECISIÓN Y VELOCIDAD ---")
    print(f"Árbol de Decisión: {acc_dt * 100:.2f}% de precisión | Tiempo: {tiempo_dt:.2f} ms")
    print(f"Random Forest:     {acc_rf * 100:.2f}% de precisión | Tiempo: {tiempo_rf:.2f} ms")
    
    # Determinar el ganador
    if acc_dt == acc_rf:
        if tiempo_dt < tiempo_rf:
            print("\nGANADOR GENERAL: Árbol de Decisión (Misma precisión, pero fue más rápido)")
        else:
            print("\nGANADOR GENERAL: Random Forest (Misma precisión, pero fue más rápido)")
    elif acc_dt > acc_rf:
        print("\nGANADOR GENERAL: Árbol de Decisión (Mayor precisión)")
    else:
        print("\nGANADOR GENERAL: Random Forest (Mayor precisión)")
    
    print("\nReporte de Clasificación (Árbol de Decisión):")
    print(classification_report(y_test, y_pred_dt, target_names=['NO (0)', 'SI (1)']))
    
    # --- ANÁLISIS DE IMPORTANCIA DE VARIABLES ---
    print("\n--- ANÁLISIS DE IMPORTANCIA DE VARIABLES ---")
    importancias = modelo_rf.feature_importances_
    variables = X.columns
    
    plt.figure(figsize=(10, 6))
    # Usamos hue=variables y legend=False para evitar warnings en nuevas versiones de seaborn
    sns.barplot(x=importancias, y=variables, hue=variables, legend=False, palette='viridis')
    plt.title('Importancia de las Variables en la Selección de Rutas (Random Forest)')
    plt.xlabel('Nivel de Importancia (0 a 1)')
    plt.ylabel('Variables del Dataset')
    
    ruta_importancia = os.path.join(os.path.dirname(__file__), '..', 'docs', 'importancia_variables.png')
    plt.savefig(ruta_importancia, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de importancia guardado exitosamente en: {os.path.abspath(ruta_importancia)}")
    
    print("\n--- REGLAS APRENDIDAS POR EL ÁRBOL ---")
    reglas = export_text(modelo_dt, feature_names=list(X.columns))
    print(reglas)
    
    # Generar y guardar la visualización gráfica del árbol
    print("\nGenerando imagen del árbol de decisión...")
    plt.figure(figsize=(15, 10))
    plot_tree(modelo_dt, 
              feature_names=list(X.columns),  
              class_names=['Descartada (NO)', 'Seleccionada (SI)'],
              filled=True, 
              rounded=True, 
              fontsize=10)
    
    ruta_imagen = os.path.join(os.path.dirname(__file__), '..', 'docs', 'arbol_decision.png')
    plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Imagen guardada exitosamente en: {os.path.abspath(ruta_imagen)}")
    
    return modelo_dt, codificadores

def ejecutar_casos_prueba(modelo, codificadores):
    print("\n" + "="*40)
    print("--- FASE DE PRUEBAS MANUALES ---")
    print("="*40)
    
    # Definimos la función para predecir
    def predecir_ruta(distancia, tiempo, trafico, transbordos, descripcion_caso):
        # Convertir tráfico a código
        trafico_cod = codificadores['trafico'][trafico]
        
        # Crear DataFrame con una fila (usamos DataFrame para evitar warnings de feature names)
        datos_prueba = pd.DataFrame({
            'distancia': [distancia],
            'tiempo': [tiempo],
            'trafico_cod': [trafico_cod],
            'transbordos': [transbordos]
        })
        
        # Predecir
        prediccion_cod = modelo.predict(datos_prueba)[0]
        resultado = "Ruta SELECCIONADA (SI)" if prediccion_cod == 1 else "Ruta DESCARTADA (NO)"
        
        print(f"\n{descripcion_caso}")
        print(f"Entrada: Tráfico={trafico}, Tiempo={tiempo}min, Distancia={distancia}km, Transbordos={transbordos}")
        print(f"Resultado de la predicción: {resultado}")
        
    # Caso 1: Bajo tráfico, tiempo corto
    predecir_ruta(distancia=10.0, tiempo=20, trafico='bajo', transbordos=0, 
                 descripcion_caso="Caso 1: Bajo tráfico y tiempo corto")
                 
    # Caso 2: Alto tráfico
    predecir_ruta(distancia=15.0, tiempo=50, trafico='alto', transbordos=1, 
                 descripcion_caso="Caso 2: Alto tráfico")
                 
    # Caso 3: Mismo tiempo, distinto número de transbordos (Evaluaremos 2 sub-casos)
    print("\nCaso 3: Comparación con mismo tiempo y distintos transbordos")
    predecir_ruta(distancia=12.0, tiempo=40, trafico='medio', transbordos=1, 
                 descripcion_caso="Sub-caso 3.1: 1 Transbordo")
    predecir_ruta(distancia=12.0, tiempo=40, trafico='medio', transbordos=3, 
                 descripcion_caso="Sub-caso 3.2: 3 Transbordos")

def menu_interactivo(modelo, codificadores):
    print("\n" + "="*50)
    print(" MENÚ INTERACTIVO: CONSULTA TU RUTA ")
    print("="*50)
    
    while True:
        try:
            print("\nIntroduce los datos de tu ruta (o escribe 'salir' para terminar):")
            
            entrada_distancia = input("Distancia en km (ej. 15.5): ")
            if entrada_distancia.lower() == 'salir': break
            distancia = float(entrada_distancia)
            
            tiempo = int(input("Tiempo estimado en minutos (ej. 45): "))
            
            trafico = input("Nivel de tráfico (bajo / medio / alto): ").lower()
            if trafico not in ['bajo', 'medio', 'alto']:
                print("Error: El tráfico debe ser 'bajo', 'medio' o 'alto'. Intenta de nuevo.")
                continue
                
            transbordos = int(input("Número de transbordos (ej. 0, 1, 2): "))
            
            # Convertir tráfico a código
            trafico_cod = codificadores['trafico'][trafico]
            
            # Crear DataFrame
            datos_prueba = pd.DataFrame({
                'distancia': [distancia],
                'tiempo': [tiempo],
                'trafico_cod': [trafico_cod],
                'transbordos': [transbordos]
            })
            
            # Predecir
            prediccion_cod = modelo.predict(datos_prueba)[0]
            
            print("\n" + "-"*40)
            if prediccion_cod == 1:
                print("RESULTADO: El modelo dice que esta es una EXCELENTE RUTA (SI).")
            else:
                print("RESULTADO: El modelo dice que DEBES DESCARTAR esta ruta (NO).")
            print("-"*40)
            
        except ValueError:
            print("Error: Por favor ingresa números válidos para distancia, tiempo y transbordos.")

if __name__ == "__main__":
    # Ejecutar flujo completo
    modelo_entrenado, codificadores = entrenar_modelo()
    
    if modelo_entrenado:
        ejecutar_casos_prueba(modelo_entrenado, codificadores)
        # Iniciar menú interactivo al final
        menu_interactivo(modelo_entrenado, codificadores)
