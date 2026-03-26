import pandas as pd
import numpy as np
import random
import os

# Configurar semilla para reproducibilidad
np.random.seed(42)
random.seed(42)

def generar_dataset(num_registros=1000):
    print(f"Generando dataset con {num_registros} registros...")
    
    # Definir posibles orígenes y destinos
    puntos = ['Centro', 'Norte', 'Sur', 'Este', 'Oeste', 'Estacion_Central', 'Aeropuerto', 'Universidad']
    
    datos = []
    
    for _ in range(num_registros):
        # Seleccionar origen y destino distintos
        origen = random.choice(puntos)
        destino = random.choice([p for p in puntos if p != origen])
        
        # Generar variables numéricas y categóricas
        distancia = round(random.uniform(2.0, 35.0), 2) # en km
        
        # El tiempo base depende de la distancia, pero añadimos ruido
        tiempo_base = distancia * 2.5 # asumiendo velocidad media
        tiempo = int(tiempo_base + random.uniform(-10, 20)) 
        tiempo = max(10, tiempo) # mínimo 10 minutos
        
        trafico = random.choice(['bajo', 'medio', 'alto'])
        
        # Ajustar tiempo según tráfico
        if trafico == 'medio':
            tiempo = int(tiempo * 1.3)
        elif trafico == 'alto':
            tiempo = int(tiempo * 1.8)
            
        transbordos = random.choice([0, 1, 2, 3])
        
        # REGLAS LÓGICAS PARA DEFINIR LA MEJOR RUTA
        # Penalizar alto tráfico, preferir tiempo menor a 45 min y max 1 transbordo
        if trafico != 'alto' and tiempo <= 45 and transbordos <= 1:
            mejor_ruta = 'SI'
        else:
            # Añadir algo de variabilidad para que el modelo tenga que aprender
            # Si el tráfico es bajo pero el tiempo es un poco más de 45, a veces puede ser la mejor ruta
            if trafico == 'bajo' and tiempo <= 60 and transbordos == 0:
                mejor_ruta = 'SI'
            else:
                mejor_ruta = 'NO'
                
        datos.append([origen, destino, distancia, tiempo, trafico, transbordos, mejor_ruta])
    
    # Crear DataFrame
    columnas = ['origen', 'destino', 'distancia', 'tiempo', 'trafico', 'transbordos', 'mejor_ruta']
    df = pd.DataFrame(datos, columns=columnas)
    
    # Mostrar distribución de la variable objetivo
    print("\nDistribución de la variable 'mejor_ruta':")
    print(df['mejor_ruta'].value_counts(normalize=True) * 100)
    
    # Guardar a CSV
    ruta_guardado = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')
    df.to_csv(ruta_guardado, index=False)
    print(f"\nDataset guardado exitosamente en: {os.path.abspath(ruta_guardado)}")

if __name__ == "__main__":
    generar_dataset()
