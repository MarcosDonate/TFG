"""
Este módulo proporciona funciones de utilidad para el preprocesamiento de datos y la
construcción de matrices de preferencias (matriz C) necesarias para los
solucionadores del problema OSBOP.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def reconstruir_matriz_desde_lista(valores):
    """
    Reconstruye una matriz de preferencias a partir de una lista de valores que
    representan el triángulo superior de la matriz.

    Args:
        valores (list): Una lista de flotantes que contiene los valores C(i, j) para i < j.

    Returns:
        np.ndarray: La matriz de preferencias C (n x n) completamente reconstruida, donde:
                    - C[i, i] = 0.5
                    - C[j, i] = 1 - C[i, j]
    """
    L = len(valores)
    if L == 0:
        return np.array([[0.5]])
    
    # El número de elementos en el triángulo superior de una matriz n x n es L=n*(n-1)/2.
    # Despejamos n para obtener el tamaño de la matriz.
    discriminante = 1 + 8 * L
    if discriminante < 0 or math.sqrt(discriminante) % 1 != 0:
        raise ValueError("La longitud de la lista no corresponde a un triángulo de matriz válido.")
    
    n = int((1 + math.sqrt(discriminante)) / 2)
    
    # Inicializa la matriz C con ceros.
    matriz_temp = np.zeros((n, n))
    k = 0

    # Rellena el triángulo superior de la matriz con los valores de la lista.
    for i in range(n):
        for j in range(n):
            if i == j:
                matriz_temp[i, j] = 0.5
            elif i < j:
                matriz_temp[i, j] = valores[k]
                k += 1

    # Rellena el triángulo inferior usando la propiedad C[j, i] = 1 - C[i, j].
    for i in range(n):
        for j in range(i + 1, n):
            matriz_temp[j, i] = 1 - matriz_temp[i, j]
            
    return matriz_temp       

def generar_matriz_conteos(datos_votacion):
    """
    Genera una matriz de comparación por pares con el recuento absoluto de votos.
    Asume que los datos de entrada son una lista de cadenas con el formato "votantes,preferido,menos_preferido".

    Args:
        datos_votacion (list): Lista de strings, donde cada string representa una votación.

    Returns:
        list: Una matriz de conteos (lista de listas) donde M[i][j] contiene el
              número de votantes que prefieren al candidato 'i' sobre el candidato 'j'.
    """
    max_candidato = 0
    # Primer paso: determinar el tamaño de la matriz encontrando el ID más alto del candidato.
    for voto in datos_votacion:
        partes = voto.split(',')
        candidato_1, candidato_2 = int(partes[1]), int(partes[2])
        max_candidato = max(max_candidato, candidato_1, candidato_2)

    # Inicializa la matriz de conteos con ceros. El tamaño es +1 para usar índices base 1.
    matriz_conteos = [[0] * (max_candidato + 1) for _ in range(max_candidato + 1)]

    # Segundo paso: rellenar la matriz con los recuentos de votos.
    for voto in datos_votacion:
        partes = voto.split(',')
        votantes, preferido, menos_preferido = int(partes[0]), int(partes[1]), int(partes[2])
        matriz_conteos[preferido][menos_preferido] += votantes
        
    return matriz_conteos

def normalizar_matriz_a_numpy(matriz_conteos):
    """
    Normaliza una matriz de conteos para crear una matriz de preferencias C.
    La matriz resultante cumple las propiedades:
    - C[i, i] = 0.5
    - C[i, j] + C[j, i] = 1

    Args:
        matriz_conteos (list): La matriz de recuentos absolutos generada por `generar_matriz_conteos`.

    Returns:
        np.ndarray: La matriz de preferencias C, normalizada y en formato NumPy.
    """
    if not matriz_conteos or len(matriz_conteos) <= 1:
        return np.array([])

    # El número de candidatos es el tamaño de la matriz menos uno (por el índice base 1).
    num_candidatos = len(matriz_conteos) - 1
    matriz_normalizada = np.zeros((num_candidatos, num_candidatos), dtype=float)

    # Itera sobre cada par de candidatos (i, j) para calcular la preferencia normalizada.
    for i in range(1, num_candidatos + 1):
        for j in range(1, num_candidatos + 1):
            if i == j:
                matriz_normalizada[i - 1, j - 1] = 0.5

            votos_i_sobre_j = matriz_conteos[i][j]
            votos_j_sobre_i = matriz_conteos[j][i]
            
            total_votos_par = votos_i_sobre_j + votos_j_sobre_i
            
            # Normaliza la preferencia de i sobre j.
            # Se evita la división por cero si no hubo votantes para este par específico.
            if total_votos_par > 0:
                matriz_normalizada[i - 1, j - 1] = votos_i_sobre_j / total_votos_par
            
    return matriz_normalizada

def crear_matriz_C_desde_votacion(datos_votacion: list) -> np.ndarray:
    """
    Función de conveniencia que realiza el proceso completo: genera la matriz de
    conteos a partir de los datos de votación y la normaliza para obtener la
    matriz de preferencias C final.

    Args:
        datos_votacion (list): Lista de strings con el formato "votantes,preferido,menos_preferido".

    Returns:
        np.ndarray: La matriz de preferencias C (n x n) lista para ser usada por los solvers.
    """
    # Paso 1: Generar la matriz de conteos a partir de los datos.
    matriz_conteos = generar_matriz_conteos(datos_votacion)

    # Paso 2: Normalizar la matriz de conteos para obtener la matriz C.
    matriz_c_final = normalizar_matriz_a_numpy(matriz_conteos)

    return matriz_c_final

def formatear_solucion_s(s_c):
    """
    Formatea un conjunto de soluciones s_c en una cadena de texto legible.

    Esta función convierte una estructura de solución anidada, como la que devuelven
    los solvers, en una cadena de texto compacta y estandarizada, ideal para
    visualización o almacenamiento.

    Args:
        s_c (list): El conjunto de soluciones (SolutionSet), que es una lista de
                    bucket orders. Cada bucket order es a su vez una lista de buckets.
                    Ejemplo: [[[1, 2], [3]], [[3, 1], [2]]]

    Returns:
        str: Una cadena de texto representando la solución formateada, donde los
             ítems en un bucket se separan por comas, los buckets por barras
             verticales, y los bucket orders por punto y coma.
             Ejemplo de salida: "1,2|3; 3,1|2"
    """
    # --- Validación de la Entrada ---
    # Si el conjunto de soluciones está vacío o no es una lista, devuelve una cadena vacía.
    if not s_c or not isinstance(s_c, list):
        return ""

    lineas_formateadas = []
    # Itera sobre cada bucket order individual en el conjunto de soluciones.
    for i, bucket_order in enumerate(s_c, start=1):
        partes_bucket = []
        # Itera sobre cada bucket (grupo de ítems en empate) dentro del bucket order.
        for bucket in bucket_order:
            # Convierte todos los ítems del bucket a string y los une con una coma.
            # Ejemplo: [1, 2] -> "1,2"
            bucket_str = ",".join(map(str, bucket))
            partes_bucket.append(bucket_str)
        
        # Une las cadenas de cada bucket con "|" para formar el string del bucket order completo.
        # Ejemplo: ["1,2", "3"] -> "1,2|3"
        orden_completa_str = "|".join(partes_bucket)
        lineas_formateadas.append(f"{orden_completa_str}")

    # Finalmente, une todos los bucket orders formateados con "; " para la salida final.
    # Ejemplo: ["1,2|3", "3,1|2"] -> "1,2|3; 3,1|2"
    return "; ".join(lineas_formateadas)

def generar_graficos(results_df, x_variable, fixed_variable_name, fixed_variable_value, output_prefix):
    """
    Genera y guarda dos gráficos a partir de un DataFrame de resultados.

    Crea un gráfico de línea para el tiempo de ejecución y otro para la calidad de la
    solución (fitness 'f'). Los gráficos muestran cómo varían estas métricas en
    función de una variable (e.g., 'n' o 'b'), manteniendo la otra constante.

    Args:
        results_df (pd.DataFrame): DataFrame con los resultados de los experimentos.
                                   Debe contener las columnas: 'algorithm', 'time', 'f',
                                   y las columnas para las variables (e.g., 'n', 'b').
        x_variable (str): El nombre de la columna que se usará en el eje X (e.g., 'n').
        fixed_variable_name (str): El nombre de la columna que se mantiene fija (e.g., 'b').
        fixed_variable_value (int): El valor de la variable que se mantiene fija.
        output_prefix (str): Prefijo para los nombres de los archivos de imagen guardados.
    """
    # --- 1. Filtrado de Datos ---
    # Selecciona solo la porción del DataFrame que coincide con el valor de la variable fija.
    plot_df = results_df[results_df[fixed_variable_name] == fixed_variable_value].copy()

    # Si el DataFrame filtrado está vacío, no hay nada que graficar.
    if plot_df.empty:
        return
    
    print(f"\n📈 Generando gráficos para '{x_variable}' variable, con {fixed_variable_name}={fixed_variable_value}...")

    # --- 2. Configuración de Estilo de Gráficos ---
    # Define una paleta de colores y un tema para asegurar una apariencia consistente.
    palette = sns.color_palette("Set1", n_colors=len(plot_df['algorithm'].unique()))
    sns.set_theme(style="whitegrid", palette=palette)

    # --- 3. Generación del Gráfico de Tiempo de Ejecución ---
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=plot_df, x=x_variable, y='time', hue='algorithm', marker='o', errorbar=('ci', 95))
    plt.title(f'Escalabilidad del Tiempo de Ejecución ({fixed_variable_name}={fixed_variable_value})', fontsize=16, weight='bold')
    plt.xlabel(f'{x_variable}', fontsize=12)
    plt.ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
    plt.legend(title='Algoritmo')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    output_filename_time = f"{output_prefix}_tiempo.png"
    plt.savefig(output_filename_time, dpi=300)
    print(f"   -> Gráfico de tiempo guardado en: {output_filename_time}")
    plt.close()

    # --- 4. Generación del Gráfico de Calidad de la Solución ---
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=plot_df, x=x_variable, y='f', hue='algorithm', marker='o', errorbar=('ci', 95))
    plt.title(f'Calidad de la Solución ({fixed_variable_name}={fixed_variable_value})', fontsize=16, weight='bold')
    plt.xlabel(f'{x_variable}', fontsize=12)
    plt.ylabel('Mejor valor f', fontsize=12)
    plt.legend(title='Algoritmo')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    output_filename_fitness = f"{output_prefix}_fitness.png"
    plt.savefig(output_filename_fitness, dpi=300)
    print(f"   -> Gráfico de fitness guardado en: {output_filename_fitness}")
    plt.close()
