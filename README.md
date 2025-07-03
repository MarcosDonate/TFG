=
        OPTIMIZACIÓN METAHEURÍSTICA PARA RESOLVER EL
        EL PROBLEMA DE AGREGACIÓN DE RANKINGS GENERALIZADO
        A UN CONJUNTO DE BUCKET ORDERS ÓPTIMOS (OSBOP)
=

Este proyecto implementa y compara tres algoritmos metaheurísticos para resolver
el Problema del Conjunto Óptimo de Bucket Orders (OSBOP). El objetivo es
encontrar un conjunto de rankings con empates que representen de la mejor
manera las preferencias de una población heterogénea.


## 1. Estructura del Proyecto

El proyecto está organizado en los siguientes archivos:

* **`OSBOP.py`**: El núcleo del proyecto. Contiene las clases que implementan los tres algoritmos de optimización: `SLSOSBOPSolver`, `EAOSBOPSolver` y `MAOSBOPSolver`.

* **`utilidades.py`**: Módulo con funciones de ayuda para tareas auxiliares, como la construcción de matrices de preferencias a partir de diferentes formatos de datos (`reconstruir_matriz_desde_lista`, `crear_matriz_C_desde_votacion`) y la generación de gráficos de resultados (`generar_graficos_escalabilidad`).

* **`datasets.json`**: Archivo de datos en formato JSON que almacena los conjuntos de datos utilizados en los experimentos. Separar los datos del código permite una gestión más sencilla y la fácil adición de nuevos datasets.

* **`experimentos.py`**: Script principal de ejecución. Implementa todo el proceso experimental: carga los datasets, configura y ejecuta los algoritmos, recopila los resultados, los procesa con pandas y los presenta en una tabla y en gráficos.


## 2. Requisitos e Instalación

Para ejecutar este proyecto, necesitas tener Python 3 instalado, junto con las siguientes librerías. Puedes instalarlas todas con un solo comando:

pip install numpy pandas rich seaborn matplotlib


## 3. ¿Cómo Ejecutar los Experimentos?

1.  **Configuración**: Asegúrate de que los cuatro archivos (`OSBOP.py`, `utilidades.py`, `experimentos.py`, `datasets.json`) estén en el mismo directorio.

2.  **Ajustar Parámetros (Opcional)**: Abre `experimentos.py` para modificar la configuración del experimento. Puedes cambiar:
    * `NUM_EJECUCIONES`: El número de veces que se repite cada experimento.
    * `num_b`: El número de bucket orders a buscar.
    * El diccionario `algoritmos`: Para ajustar los hiperparámetros de cada metaheurística (tamaño de la población, número de generaciones, etc.).

3.  **Ejecución**: Abre una terminal en el directorio del proyecto y ejecuta el script:

    python experimentos.py


## 4. ¿Qué Resultados se Obtienen?

Al finalizar la ejecución, el script generará dos tipos de salida:

1.  **En la Consola**: Se imprimirá una tabla resumen, generada con la librería `rich`, que muestra las estadísticas de rendimiento (media de fitness, desviación estándar, mejor resultado, tiempos de ejecución, etc.) para cada algoritmo en cada dataset procesado.

2.  **Archivos de Gráficos**: Se guardarán imágenes en formato `.png` con los gráficos de escalabilidad. Por defecto, se guardarán en el directorio 'gráficas'.
