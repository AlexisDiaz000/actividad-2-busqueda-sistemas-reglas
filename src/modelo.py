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
    
    return modelo_dt, modelo_rf, acc_dt, acc_rf, codificadores

def menu_interactivo(modelo_dt, modelo_rf, acc_dt, acc_rf, codificadores):
    print("\n" + "="*50)
    print(" MENÚ INTERACTIVO: COMPARATIVA DE MODELOS SUPERVISADOS ")
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
            
            print("\nSelecciona el modelo que deseas probar:")
            print("1. Árbol de Decisión (Decision Tree)")
            print("2. Bosque Aleatorio (Random Forest)")
            print("3. Comparar ambos al mismo tiempo")
            
            seleccion_modelo = input("Ingresa el número (1, 2 o 3): ")
            if seleccion_modelo not in ['1', '2', '3']:
                print("Error: Por favor ingresa 1, 2 o 3.")
                continue
            
            # Convertir tráfico a código
            trafico_cod = codificadores['trafico'][trafico]
            
            # Crear DataFrame
            datos_prueba = pd.DataFrame({
                'distancia': [distancia],
                'tiempo': [tiempo],
                'trafico_cod': [trafico_cod],
                'transbordos': [transbordos]
            })
            
            print("\n" + "-"*50)
            
            if seleccion_modelo in ['1', '3']:
                # Predicción con Decision Tree
                inicio_pred_dt = time.time()
                pred_dt = modelo_dt.predict(datos_prueba)[0]
                tiempo_pred_dt = (time.time() - inicio_pred_dt) * 1000
                
                print("🌳 ÁRBOL DE DECISIÓN:")
                print(f"Precisión global del modelo: {acc_dt * 100:.2f}%")
                print(f"Velocidad de respuesta:      {tiempo_pred_dt:.4f} ms")
                if pred_dt == 1:
                    print("Decisión de Ruta:            ✅ SELECCIONADA (SI)")
                else:
                    print("Decisión de Ruta:            ❌ DESCARTADA (NO)")
                print("-" * 50)
                
            if seleccion_modelo in ['2', '3']:
                # Predicción con Random Forest
                inicio_pred_rf = time.time()
                pred_rf = modelo_rf.predict(datos_prueba)[0]
                tiempo_pred_rf = (time.time() - inicio_pred_rf) * 1000
                
                print("🌲 BOSQUE ALEATORIO (Random Forest):")
                print(f"Precisión global del modelo: {acc_rf * 100:.2f}%")
                print(f"Velocidad de respuesta:      {tiempo_pred_rf:.4f} ms")
                if pred_rf == 1:
                    print("Decisión de Ruta:            ✅ SELECCIONADA (SI)")
                else:
                    print("Decisión de Ruta:            ❌ DESCARTADA (NO)")
                print("-" * 50)
            
        except ValueError:
            print("Error: Por favor ingresa números válidos para distancia, tiempo y transbordos.")

if __name__ == "__main__":
    # Ejecutar flujo completo de entrenamiento
    modelo_dt, modelo_rf, acc_dt, acc_rf, codificadores = entrenar_modelo()
    
    if modelo_dt and modelo_rf:
        # Iniciar menú interactivo directamente
        menu_interactivo(modelo_dt, modelo_rf, acc_dt, acc_rf, codificadores)
