"""
Script principal para la ejecución y evaluación de los algoritmos de OSBOP.

Este script realiza el siguiente proceso:
1. Carga los datasets desde un archivo JSON.
2. Define los parámetros para cada algoritmo a probar.
3. Itera sobre cada dataset y cada algoritmo, ejecutándolos un número
   definido de veces con diferentes semillas aleatorias.
4. Recopila los resultados de cada ejecución (f_C, tiempo, solución, etc.).
5. Procesa los resultados usando pandas para calcular estadísticas agregadas
   (media, desviación estándar) y encontrar la mejor ejecución.
6. Muestra una tabla de resumen en la consola y genera gráficos de escalabilidad.
"""
import time
import numpy as np
import json
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# Importa los módulos con los solucionadores y las funciones de utilidad.
import OSBOP
import utilidades as util


# --- 1. CONFIGURACIÓN DEL EXPERIMENTO ---

# Carga la definición de los datasets desde el archivo JSON.
with open('datos/datasets.json', 'r') as f:
    datasets = json.load(f)

# Define cuántas veces se ejecutará cada algoritmo por dataset.
NUM_EJECUCIONES = 5
# Define el número de bucket orders (parámetro 'b') para esta tanda de pruebas.
num_b=2

# Genera un conjunto fijo de semillas aleatorias para asegurar que todas las
# comparaciones entre algoritmos sean justas y los resultados reproducibles.
master_rng = np.random.default_rng(42)
execution_seeds = [master_rng.integers(0, 10000) for _ in range(NUM_EJECUCIONES)]

# Listas para almacenar los resultados de todas las ejecuciones.
all_run_results = []

# --- 2. BUCLE PRINCIPAL DE EJECUCIÓN ---

# Itera sobre cada uno de los datasets definidos en el archivo JSON.
for dataset_name, dataset_data in datasets['datasets_little'].items():
    print(f"======================================================")
    print(f"▶️  Procesando Dataset: '{dataset_name}'")
    print(f"======================================================")

    # Construye la matriz de preferencias C a partir de los datos del dataset.
    if dataset_data['tipo'] == 'matriz_superior':
        C_matrix = util.reconstruir_matriz_desde_lista(dataset_data['datos'])
    elif dataset_data['tipo'] == 'votacion':
        C_matrix = util.crear_matriz_C_desde_votacion(dataset_data['datos'])
    else:
        print(f"❌ Tipo de dataset no reconocido: {dataset_data['tipo']}")
        break
    n_items = C_matrix.shape[0]

    # Diccionario para configurar los algoritmos y sus hiperparámetros.
    algoritmos = {
        "SLSOSBOP": {
            "solver_class": OSBOP.SLSOSBOPSolver,
            "params": {
                "b_num_bucket_orders": num_b,
                "equal_weights_eq": False,
                "t1_outer_iterations": 10000,
                "t2_inner_iterations": 100
            }
        },
        "EAOSBOP": {
            "solver_class": OSBOP.EAOSBOPSolver,
            "params": {
                "b_num_bucket_orders": num_b,
                "population_size": 150,
                "num_generations": 250,
                "crossover_prob": 0.8,
                "mutation_prob_bo": 0.2,
                "mutation_prob_w": 0.1,
                "tournament_size": 3,
                "elitism_count": 1
            }
        },
        "MAOSBOP": {
            "solver_class": OSBOP.MAOSBOPSolver,
            "params": {
                "b_num_bucket_orders": num_b,
                "population_size": 100,
                "num_generations": 150,
                "crossover_prob": 0.8,
                "mutation_prob_bo": 0.2,
                "mutation_prob_w": 0.1,
                "local_search_prob": 0.5,
                "t_ls_weights": 50,
                "tournament_size": 3,
                "elitism_count": 1,
                "ble_max_steps": 20
            }
        }
    }      

    # Itera sobre cada algoritmo configurado para probarlo en el dataset actual.
    for algo_name, algo_info in algoritmos.items():
        print(f"\n--- Ejecutando {algo_name} ---")

        best_run_f = float('inf')
        best_run_s = None
        best_run_w = None
        best_run_time = None

        # Bucle para realizar el número de ejecuciones definidas y promediar resultados.
        for i in range(NUM_EJECUCIONES):
            current_seed = execution_seeds[i]
            print(f"Ejecución {i + 1}/{NUM_EJECUCIONES}...")

            start_time = time.time()

            solver_params = algo_info["params"].copy()
            solver_params["seed"] = current_seed

            # Instancia el solver correspondiente con sus parámetros y la semilla actual.
            solver = algo_info["solver_class"](
                C_matrix, **algo_info["params"]
            )

            s, w, f = solver.solve()

            end_time = time.time()
            execution_time = end_time - start_time

            print(f"Resultado: f={f}, s={s}, w={w}, Tiempo: {execution_time:.2f} segundos")

            if f < best_run_f:
                best_run_f = f
                best_run_s = s
                best_run_w = w
                best_run_time = execution_time

            # Almacena los resultados de esta ejecución en un diccionario
            all_run_results.append({
                "dataset": dataset_name,
                "algorithm": algo_name,
                "run": i + 1,
                "n": n_items,
                "b": num_b,
                "f": f,
                "s": s,
                "w": w,
                "time": execution_time,
            })

# --- 3. PROCESAMIENTO Y VISUALIZACIÓN DE RESULTADOS ---

print("\n\n✅ Todas las ejecuciones completadas.")
print("======================================================")
print("      MEJORES RESULTADOS CONSOLIDADOS (TABLA)         ")
print("======================================================")

if not all_run_results:
    print("No se generaron resultados.")
else:
    # Convierte la lista de resultados en un DataFrame de pandas para un fácil análisis
    all_results_df = pd.DataFrame(all_run_results)

    # --- Cálculo de Estadísticas ---
    # Agrupa por dataset y algoritmo para calcular la media y desviación estándar
    agg_stats = all_results_df.groupby(['dataset', 'algorithm']).agg(
        mean_f=('f', 'mean'),
        std_f=('f', 'std'),
        mean_time=('time', 'mean'),
        std_time=('time', 'std'),
    ).reset_index()

    best_run_indices = all_results_df.groupby(['dataset', 'algorithm'])['f'].idxmin()
    best_run_info = all_results_df.loc[best_run_indices, ['dataset', 'algorithm', 'f', 's', 'w', 'time']].copy()

    best_run_info['best_s_formatted'] = best_run_info['s'].apply(util.formatear_solucion_s)

    best_run_info.rename(columns={
        'f': 'best_f',
        'w': 'best_w',
        'time': 'time_for_best_f'
    }, inplace=True)

    summary_df = pd.merge(
        agg_stats, 
        best_run_info[['dataset', 'algorithm', 'best_f', 'best_s_formatted', 'best_w', 'time_for_best_f']],
        on=['dataset', 'algorithm']
    )

    console_display_df = summary_df[[
        "dataset",
        "algorithm",
        "mean_f",
        "std_f",
        "mean_time",
        "std_time",
        "best_f",
        "time_for_best_f",
        "best_s_formatted",
        "best_w"
    ]].copy()

    console_display_df.rename(columns={
        "dataset": "Dataset",
        "algorithm": "Algoritmo",
        "mean_f": "Media (f)",
        "std_f": "DE (f)",
        "mean_time": "Tiempo Medio (s)",
        "std_time": "DE Tiempo (s)",
        "best_f": "Mejor (f)",
        "time_for_best_f": "Tiempo Mejor (s)",
        "best_s_formatted": "Mejor Solución (s)",
        "best_w": "Mejor (w)"
    }, inplace=True)

    pd.options.display.float_format = '{:,.4f}'.format
    for col in console_display_df.columns:
        if console_display_df[col].dtype == 'float64':
            console_display_df[col] = console_display_df[col].round(4)

    pd.set_option('display.max_colwidth', None)
    console_display_df.fillna(0, inplace=True)

    # Usa la librería 'rich' para imprimir una tabla en la consola.
    console = Console()
    table = Table(
        title="Resultados por Algoritmo y Dataset",
        box=box.MINIMAL_DOUBLE_HEAD,
        header_style="bold cyan",
        show_header=True
    )

    for column in console_display_df.columns:
        table.add_column(column, justify="left")

    for index, row in console_display_df.iterrows():
        table.add_row(
            *[str(row[col]) for col in console_display_df.columns]
        )

    console.print(table)

    # --- Generación de Gráficos ---
    # Llama a la función de utilidad para generar los gráficos de escalabilidad.
    util.generar_graficos(
        results_df=all_results_df,
        x_variable='n',
        fixed_variable_name='b',
        fixed_variable_value=2,
        output_prefix='graficas/grafico_n'
    )