"""
Este módulo conitene la implementación de tres algoritmos metaheurísticos para resolver el 
Optimal Set of Bucket Orders Problem (OSBOP).

Clases implementadas:
- SLSOSBOPSolver: Implementa un algoritmo basado en Búsqueda Local Estocástica (SLS) para 
resolver el OSBOP.
- EAOSBOPSolver: Implementa un Algoritmo Evolutivo (EA) para resolver el OSBOP.
- MAOSBOPSolver: Implementa un Algoritmo Memético (MA) para resolver el OSBOP, que hibrida el
algoritmo de Algortimo Evolutivo con Búsqueda Local.
"""

import random
import numpy as np
import math
import copy
from typing import List, Tuple, Any

# --- Definición de Tipos para Claridad ---
Item = Any # Un ítem puede ser cualquier tipo, como int o str.
Bucket = List[Item]  # Un bucket es una lista de ítems.
BucketOrder = List # Un bucket order es una lista ordenada de buckets.
SolutionSet = List # Un conjunto de solución es una lista de Bucket Orders.
WeightVector = List[float] # Un vector de pesos para el conjunto de soluciones.
PrecedenceMatrix = np.ndarray # La matriz de precedencia C.

class SLSOSBOPSolver:
    """
    Implementa el algoritmo de Búsqueda Local Estocástica (SLS) para el problema OSBOP.

    Este algoritmo explora el espacio de soluciones partiendo de una solución aleatoria y
    aplicando iterativamente operadores de mutación para encontrar una solución de mejor calidad.
    """

    def __init__(self, C_matrix: PrecedenceMatrix, b_num_bucket_orders: int, equal_weights_eq: bool, t1_outer_iterations: int, t2_inner_iterations: int = 100, seed: int = None):
        """
        Inicializa el solucionador SLS-OSBOP.

        Args:
            C_matrix (PrecedenceMatrix): Matriz de precedencia (n x n) que define el problema.
            b_num_bucket_orders (int): Número de bucket orders a encontrar en la solución (parámetro 'b').
            equal_weights_eq (bool): Si es True, todos los bucket orders tendrán un peso igual (1/b).
                                     Si es False, los pesos se optimizarán.
            t1_outer_iterations (int): Número de iteraciones para el bucle de búsqueda principal.
            t2_inner_iterations (int): Número de iteraciones para el ajuste fino de pesos.
            seed (int, optional): Semilla para la generación de números aleatorios.
        """

        if seed is not None:
            random.seed(seed)

        # --- Validación de Parámetros de entrada ---
        if not isinstance(C_matrix, np.ndarray) or C_matrix.ndim != 2 or C_matrix.shape[0] != C_matrix.shape[1]:
            raise ValueError("C_matrix debe ser una matriz cuadrada de numpy.ndarray.")
        if b_num_bucket_orders <= 0:
            raise ValueError("b_num_bucket_orders debe ser un entero positivo.")
        if t1_outer_iterations <= 0 or t2_inner_iterations <= 0:
            raise ValueError("Los contadores de iteración (t1, t2) deben ser enteros positivos.")
        
        self.C_matrix = C_matrix
        self.num_items = C_matrix.shape[0]
        self.items_list = list(range(self.num_items))

        self.b = b_num_bucket_orders
        self.Eq = equal_weights_eq
        self.t1 = t1_outer_iterations
        self.t2 = t2_inner_iterations

    def _generate_random_bucket_order(self) -> BucketOrder:
        """Genera un único bucket order aleatorio."""
        shuffled_items = list(self.items_list)
        random.shuffle(shuffled_items)

        # Casos base para 0 o 1 ítems
        if self.num_items == 0:
            return []  
        if self.num_items == 1:
            return [shuffled_items]
        
        # Se crea un número aleatorio de buckets entre 1 y num_items.
        num_buckets = random.randint(1, self.num_items)
        bo: BucketOrder = [[] for _ in range(num_buckets)]

        # Se asegura que cada bucket tenga al menos un ítem al principio.
        for i in range(min(self.num_items, num_buckets)):
            bo[i].append(shuffled_items.pop())

        # Distribuye los ítems restantes aleatoriamente entre los buckets.
        while shuffled_items:
            chosen_bucket_idx = random.randrange(num_buckets)
            bo[chosen_bucket_idx].append(shuffled_items.pop())

        # Elimina buckets que puedan haber quedado vacíos y maneja casos extremos.
        bo = [ b for b in bo if b]
        if not bo and self.num_items > 0:
            return [list(self.items_list)] # Devuelve un solo bucket con todos los ítems.
        return bo
    
    def _initial_solution(self) -> SolutionSet:
        """Genera una solución aleatoria compuesta por b bucket orders."""
        s_c: SolutionSet = []
        for _ in range(self.b):
            s_c.append(self._generate_random_bucket_order())
        return s_c

    def _bucket_order_to_matrix(self, bucket_order: BucketOrder) -> np.ndarray:
        """Convierte un bucket order (lista de listas) a su representación matricial B."""
        B = np.full((self.num_items, self.num_items), 0.5, dtype=float)

        item_to_bucket_idx = {}
        for b_idx, bucket_content in enumerate(bucket_order):
            for item in bucket_content:
                item_to_bucket_idx[item] = b_idx

        for u in range(self.num_items):
            for v in range(self.num_items):
                if u == v:
                    continue # B[u, v] = 0.5

                u_b_idx = item_to_bucket_idx.get(u)
                v_b_idx = item_to_bucket_idx.get(v)

                if u_b_idx is None or v_b_idx is None:
                    # Esto implica que un item no está en ningún bucket
                    # lo que inidica un problema con la generación o mutación del bucket order
                    # Trataremos el item como si estuvieran en un bucket separado
                    # Si esto sucede, B[u, v] = 0.5
                    pass
                elif u_b_idx < v_b_idx:
                    B[u, v] = 1.0
                    B[v, u] = 0.0
                elif u_b_idx > v_b_idx:
                    B[u, v] = 0.0
                    B[v, u] = 1.0
        
        return B
    
    def _evaluate_solution(self, s_c: SolutionSet, w_c: WeightVector) -> float:
        """
        Evalúa una solución (s_c, w_c) calculando la distancia L1 entre la matriz
        ponderada de la solución (B_bar) y la matriz de entrada C.
        """
        B_bar = np.zeros((self.num_items, self.num_items), dtype=float)
        for k in range(self.b):
            B_k = self._bucket_order_to_matrix(s_c[k])
            B_bar += w_c[k] * B_k
        
        distance = np.sum(np.abs(B_bar - self.C_matrix))
        return distance
    
    def _tune_weights(self, s_c: SolutionSet, initial_w_c: WeightVector, initial_f_c: float) -> Tuple[float, WeightVector]:
        """Ajusta los pesos w_c para un s_c fijo para minimizar la función objetivo (distancia)."""
        current_w_c = list(initial_w_c)
        current_f_c = initial_f_c

        for _ in range(self.t2):
            candidate_w_c = list(current_w_c)

            if self.b == 0: break # No hay pesos para ajustar.
            if self.b == 1: # Solo un peso, debe ser 1.0
                candidate_w_c = [1.0]
            else:
                # Selecciona un peso para modificar
                idx_to_change = random.randrange(self.b)
                r = random.uniform(-0.5, 0.5)
                old_w_k_val = candidate_w_c[idx_to_change]
                new_w_k_val = max(0.01, min(1.0, old_w_k_val + r)) 
                candidate_w_c[idx_to_change] = new_w_k_val

                sum_others_old = 0
                for i in range(self.b):
                    if i!= idx_to_change:
                        sum_others_old += candidate_w_c[i]

                sum_others_new_target = 1.0 - new_w_k_val

                # Si los otros pesos son todos cero, asigna el nuevo valor a todos los demás.
                if sum_others_old == 0:
                    if sum_others_new_target > 0 and self.b > 1:
                        val_for_others = sum_others_new_target / (self.b - 1)
                        for i in range(self.b):
                            if i != idx_to_change:
                                candidate_w_c[i] = val_for_others
                else:
                    scale_factor = sum_others_new_target / sum_others_old
                    for i in range(self.b):
                        if i != idx_to_change:
                            candidate_w_c[i] = current_w_c[i] * scale_factor

            # Normalización final para corregir posibles imprecisiones.
            for i in range(self.b):
                candidate_w_c[i] = max(0.01, min(1.0, candidate_w_c[i]))

            current_sum = sum(candidate_w_c)
            if current_sum > 0:
                candidate_w_c = [w / current_sum for w in candidate_w_c]
            elif self.b > 0:
                candidate_w_c = [1.0 /self.b] * self.b

            f_c_candidate = self._evaluate_solution(s_c, candidate_w_c)

            # Acepta la nueva solución si mejora la solución.
            if f_c_candidate < current_f_c:
                current_f_c = f_c_candidate
                current_w_c = candidate_w_c

        return current_f_c, current_w_c

    def _apply_specific_mutation(self, bo: BucketOrder, mutation_type: str) -> BucketOrder:
        """Aplica uno de los siete operadores de mutación definidos a un bucket order."""
        bo_copy = [list(b) for b in bo]  # Copia profunda del bucket order original.

        # Mutación 1: Inserta un bucket en una nueva posición aleatoria.
        if mutation_type == "bucket_insertion":
            if len(bo) < 2: return bo
            idx_to_move = random.randrange(len(bo_copy))
            bucket_val = bo_copy.pop(idx_to_move)
            new_idx = random.randrange(len(bo_copy) + 1)
            bo_copy.insert(new_idx, bucket_val)
            return bo_copy
        
        # Mutación 2: Intercambia las posiciones de dos buckets.
        elif mutation_type == "bucket_interchange":
            if len(bo) < 2: return bo
            idx1, idx2 = random.sample(range(len(bo_copy)), 2)
            bo_copy[idx1], bo_copy[idx2] = bo_copy[idx2], bo_copy[idx1]
            return bo_copy
        
        # Mutación 3: Invierte el orden de una subsecuencia de buckets.
        elif mutation_type == "bucket_inversion":
            if len(bo) < 2: return bo 
            start = random.randrange(len(bo_copy))
            end = random.randrange(start, len(bo_copy))
            sub_list = bo_copy[start:end+1]
            sub_list.reverse()
            bo_copy[start:end+1] = sub_list
            return bo_copy
        
        # Mutación 4: Une dos buckets consecutivos en uno solo.
        elif mutation_type == "bucket_union":
            if len(bo) < 2: return bo
            idx_to_join = random.randrange(len(bo_copy) - 1)
            bucket1 = bo_copy.pop(idx_to_join)
            bo_copy[idx_to_join] = bucket1 + bo_copy[idx_to_join]
            return bo_copy
        
        # Mutación 5: Divide un bucket con dos o más ítems en dos buckets nuevos.
        elif mutation_type == "bucket_division":
            eligible_indices = [i for i, bucket in enumerate(bo_copy) if len(bucket) >= 2]
            if not eligible_indices: return bo
            idx_to_split = random.choice(eligible_indices)
            bucket_to_split = bo_copy[idx_to_split]
            shuffled_items = list(bucket_to_split)
            random.shuffle(shuffled_items)
            split_point = random.randint(1, len(shuffled_items) - 1)
            new_b1 = shuffled_items[:split_point]
            new_b2 = shuffled_items[split_point:]
            bo_copy[idx_to_split] = new_b1
            bo_copy.insert(idx_to_split + 1, new_b2)
            bo_copy = [b for b in bo_copy if b]
            return bo_copy
        
        # Mutación 6: Mueve un ítem de un bucket a otro bucket aleatorio o crea un nuevo bucket.
        elif mutation_type == "item_insertion":
            # Encuentra un ítem para mover.
            item_locations = []
            for b_idx, bucket_content in enumerate(bo_copy):
                for i_idx, item_val in enumerate(bucket_content):
                    item_locations.append({'item': item_val, 'b_idx': b_idx, 'i_idx': i_idx})

            if not item_locations: return bo
            chosen_item_loc = random.choice(item_locations)
            item_to_move = chosen_item_loc['item']
            b_idx_orig = chosen_item_loc['b_idx']
            original_bucket_content = bo_copy[b_idx_orig]
            original_bucket_content.remove(item_to_move)

            # Decide dónde insertar el ítem.
            if random.random() < 0.5 and bo_copy: # Insertart en un bucket existente.
                b_idx_dest = random.randrange(len(bo_copy))
                i_idx_dest = random.randrange(len(bo_copy[b_idx_dest]) + 1)
                bo_copy[b_idx_dest].insert(i_idx_dest, item_to_move)
            else: # Crear un nuevo bucket (singleton) para el ítem .
                new_singleton = [item_to_move]
                new_b_idx_for_singleton = random.randrange(len(bo_copy) + 1)
                bo_copy.insert(new_b_idx_for_singleton, new_singleton)

            bo_copy = [b for b in bo_copy if b]  # Elimina buckets vacíos.
            return bo_copy
        
        # Mutación 7: Intercambia dos ítems entre dos buckets diferentes.
        elif mutation_type == "item_interchange":
            non_empty_bucket_indices = [i for i, b_val in enumerate(bo_copy) if b_val]
            if len(non_empty_bucket_indices) < 2: return bo
            b_idx1, b_idx2 = random.sample(non_empty_bucket_indices, 2)
            i_idx1 = random.randrange(len(bo_copy[b_idx1]))
            i_idx2 = random.randrange(len(bo_copy[b_idx2]))
            item1 = bo_copy[b_idx1][i_idx1]
            item2 = bo_copy[b_idx2][i_idx2]
            bo_copy[b_idx1][i_idx1] = item2
            bo_copy[b_idx2][i_idx2] = item1
            return bo_copy
        
        return bo  # Si no se reconoce el tipo de mutación, devuelve el bucket order original.
    
    def _mutate_solution_set(self, current_s_c: SolutionSet) -> SolutionSet:
        """Aplica una mutación a un bucket order aleatorio dentro del conjunto de soluciones."""
        mutation_types = [
            "bucket_insertion",
            "bucket_interchange",
            "bucket_inversion",
            "bucket_union",
            "bucket_division",
            "item_insertion",
            "item_interchange"
        ]

        mutated_s_c = [[list(b) for b in bo] for bo in current_s_c]

        num_bo_to_mutate_this_step = 1 # Número de bucket orders a mutar en esta iteración (random.randint(1, self.b)).

        # Elige un bucket order del conjunto de soluciones para mutar.
        indices_to_mutate = random.sample(range(self.b), num_bo_to_mutate_this_step)

        for idx_bo_to_mutate in indices_to_mutate:
            bo_to_mutate = mutated_s_c[idx_bo_to_mutate]
            
            # Elige y aplica una de las mutaciones.
            chosen_mutation_type = random.choice(mutation_types)
            mutated_bo = self._apply_specific_mutation(bo_to_mutate, chosen_mutation_type)
            mutated_s_c[idx_bo_to_mutate] = mutated_bo

        return mutated_s_c
    
    def solve(self) -> Tuple:
        """
        Ejecuta el algoritmo SLS-OSBOP.

        Returns:
            Tuple[SolutionSet, WeightVector, float]: La mejor solución encontrada,
            compuesta por el conjunto de bucket orders, sus pesos y su valor de fitness.
        """
        # --- 1. Inicialización ---
        s_c = self._initial_solution()
        w_c = [1.0 / self.b] * self.b if self.b > 0 else [1.0]
        f_c= self._evaluate_solution(s_c, w_c)

        # Ajuste inicial de pesos si está habilitado.
        if not self.Eq and self.b > 0:
            f_c, w_c = self._tune_weights(s_c, w_c, f_c)

        best_s_c = [[list(b) for b in bo] for bo in s_c]
        best_w_c = list(w_c)
        best_f_c = f_c

        # --- 2. Bucle de búsqueda principal ---
        for i in range(self.t1):
            # Genera una solución vecina mutando la mejor solución actual.
            s_c_prime = self._mutate_solution_set(best_s_c)
            w_c_prime = [1.0 / self.b] * self.b if self.b > 0 else [1.0]
            f_c_prime = self._evaluate_solution(s_c_prime, w_c_prime)

            # Optimiza los pesos para la nueva estructura si es necesario.
            if not self.Eq and self.b > 0:
                f_c_prime, w_c_prime = self._tune_weights(s_c_prime, w_c_prime, f_c_prime)

            # --- 3. Criterio de aceptación ---
            # Se acepta la nueva solución si es mejor o igual que la actual.
            if f_c_prime <= best_f_c:
                best_s_c = s_c_prime
                best_w_c = w_c_prime
                best_f_c = f_c_prime
            
        return best_s_c, best_w_c, best_f_c

# --- Clases y Solucionadores para Algoritmos Poblacionales (EA y MA) ---

class Individual:
    def __init__(self, bucket_orders: SolutionSet, weights: WeightVector):
        """
        Representa a un individuo en una población, encapsulando una solución completa
        (un conjunto de bucket orders y sus pesos) y su valor de fitness.
        """
        self.bucket_orders: SolutionSet = bucket_orders
        self.weights: WeightVector = weights
        self.fitness: float = float('inf')

    def __repr__(self) -> str:
        """Representación en cadena para una fácil depuración."""
        bo_strs =[]
        for bo in self.bucket_orders:
            if len(str(bo)) > 50:
                bo_strs.append(str(bo)[:47]+"...")
            else:
                bo_strs.append(str(bo))
        return (f"Ind(Fit: {self.fitness:.4f}", 
                f"W: {[round(w, 3) for w in self.weights]})",
                f"BOs: {bo_strs})")
    
    def __lt__(self, other: 'Individual') -> bool:
        """Permite la comparación entre individuos basada en el fitness (para minimización)."""
        return self.fitness < other.fitness
    
    def clone(self) -> 'Individual':
        """Crea una copia profunda del individuo para evitar la modificación por referencia."""
        bo_copy = [[list(b) for b in bo] for bo in self.bucket_orders]
        w_copy = list(self.weights)
        cloned_ind = Individual(bo_copy, w_copy)
        cloned_ind.fitness = self.fitness
        return cloned_ind

class EAOSBOPSolver:
    """
    Implementa un Algoritmo Evolutivo (EA) para el problema OSBOP.

    Mantiene una población de soluciones que evoluciona a lo largo de generaciones
    mediante selección, cruce y mutación para encontrar una solución óptima.
    """
    def __init__(self, C_matrix: PrecedenceMatrix, b_num_bucket_orders: int, population_size:int,
                 num_generations: int, crossover_prob: float, mutation_prob_bo: float, mutation_prob_w: float,
                 tournament_size: int = 3, elitism_count: int = 1, seed: int = None):
        
        if seed is not None:
            random.seed(seed)
        
        self.C_matrix = C_matrix
        self.num_items = C_matrix.shape[0]
        self.b = b_num_bucket_orders
        self.population_size = population_size # Tamaño de la población.
        self.num_generations = num_generations # Número de generaciones.
        self.p_crossover = crossover_prob # Probabilidad de realizar cruce entre individuos.
        self.p_mutation_bo = mutation_prob_bo # Probabilidad de mutación de bucket orders.
        self.p_mutation_w = mutation_prob_w # Probabilidad de mutación de pesos.
        self.tournament_size = tournament_size # Tamaño del torneo para la selección de padres.
        self.elitism_count = elitism_count # Número de individuos a mantener sin cambios en cada generación.

        # Se reutiliza el SLSOSBOPSolver como una clase de utilidad para operaciones comunes.
        self.sls_utility_solver = SLSOSBOPSolver(
            C_matrix=self.C_matrix,
            b_num_bucket_orders=self.b,
            equal_weights_eq=False,  # En EA-OSBOP, los pesos no son iguales.
            t1_outer_iterations=1,  # Valor por defecto, no se usa en EA-OSBOP.
            t2_inner_iterations=1,   # Valor por defecto, no se usa en EA-OSBOP.
            seed=seed
        )
        self.population: List[Individual] = []

    def _normalize_weights_inplace(self, weights: WeightVector):
        """Normaliza un vector de pesos para que sus componentes sumen 1 y no sean negativos."""
        if not weights: return 

        # Asegurarse de que los pesos sean no negativos.
        for i in range(len(weights)):
            weights[i] = max(0.01, weights[i])

        current_sum = sum(weights)
        if current_sum > 0:
            for i in range(len(weights)):
                weights[i] /= current_sum
        elif len(weights) > 0: # Si todos los pesos son cero, asignar pesos iguales.
            for i in range(len(weights)):
                weights[i] = 1.0 / len(weights)

    def _initialize_population(self):
        """Crea la población inicial con individuos aleatorios."""
        self.population = []
        for _ in range(self.population_size):
            s_c: SolutionSet = [self.sls_utility_solver._generate_random_bucket_order() for _ in range(self.b)]
            w_c: WeightVector = [1.0 / self.b] * self.b if self.b > 0 else [1.0]
            individual = Individual(s_c, w_c)
            individual.fitness = self.sls_utility_solver._evaluate_solution(individual.bucket_orders, individual.weights)
            self.population.append(individual)

    def _tournament_selection(self) -> Individual:
        """Selecciona un individuo de la población mediante un torneo."""
        tournament = random.sample(self.population, self.tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)
    
    def _crossover_individuals(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Realiza un cruce híbrido entre dos padres para generar dos descendientes."""
        # Cruce uniforme para las estructuras de bucket orders.
        offspring1_bo = [[list(b_item) for b_item in bo] for bo in parent1.bucket_orders]
        offspring1_w = list(parent1.weights)
        offspring2_bo = [[list(b_item) for b_item in bo] for bo in parent2.bucket_orders]
        offspring2_w = list(parent2.weights)

        for k in range(self.b):
            if random.random() < 0.5:
                offspring1_bo[k], offspring2_bo[k] = offspring2_bo[k], offspring1_bo[k]

        #Cruce aritmético (BLX-alpha) para los pesos.
        if self.b > 0:
            alpha = random.random()
            for k in range(self.b):
                p1_wk = parent1.weights[k]
                p2_wk = parent2.weights[k]
                offspring1_w[k] = alpha * p1_wk + (1 - alpha) * p2_wk
                offspring2_w[k] = (1 - alpha) * p1_wk + alpha * p2_wk
            self._normalize_weights_inplace(offspring1_w)
            self._normalize_weights_inplace(offspring2_w)
        
        return Individual(offspring1_bo, offspring1_w), Individual(offspring2_bo, offspring2_w)
    
    def _mutate_individual(self, individual: Individual):
        """Aplica mutación a un individuo (tanto a la estructura como a los pesos)."""
        mutation_types = [
            "bucket_insertion",
            "bucket_interchange",
            "bucket_inversion",
            "bucket_union",
            "bucket_division",
            "item_insertion",
            "item_interchange"
        ]
        # Mutación estructural.
        for k in range(self.b):
            if random.random() < self.p_mutation_bo:
                chosen_mutation_type = random.choice(mutation_types)
                individual.bucket_orders[k] = self.sls_utility_solver._apply_specific_mutation(individual.bucket_orders[k], chosen_mutation_type)

        # Mutación de pesos.
        if self.b > 0:
            weights_mutated = False
            for k in range(self.b):
                if random.random() < self.p_mutation_w:
                    perturbation = random.gauss(0, 0.1)  # Perturbación gaussiana
                    individual.weights[k] += perturbation
                    weights_mutated = True

            if weights_mutated:
                self._normalize_weights_inplace(individual.weights)

    def solve(self) -> Tuple:
        """Ejecuta el algoritmo evolutivo."""
        self._initialize_population()
        best_overall_individual = min(self.population, key=lambda ind: ind.fitness).clone()

        for generation in range(self.num_generations):
            new_population: List[Individual] = []

            # Elitismo: los mejores individuos pasan directamente a la siguiente generación.
            if self.elitism_count > 0:
                self.population.sort()
                elites = [ind.clone() for ind in self.population[:self.elitism_count]]
                new_population.extend(elites)

            # Rellena el resto de la población con descendencia.
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                if random.random() < self.p_crossover:
                    offspring1, offspring2 = self._crossover_individuals(parent1, parent2)
                else:
                    offspring1 = parent1.clone()
                    offspring2 = parent2.clone()

                self._mutate_individual(offspring1)
                self._mutate_individual(offspring2)

                # Evalua y añade la nueva solución.
                offspring1.fitness = self.sls_utility_solver._evaluate_solution(offspring1.bucket_orders, offspring1.weights)
                new_population.append(offspring1)
                if len(new_population) >= self.population_size:
                    break

                offspring2.fitness = self.sls_utility_solver._evaluate_solution(offspring2.bucket_orders, offspring2.weights)
                new_population.append(offspring2)
                if len(new_population) >= self.population_size:
                    break

            self.population = new_population[:self.population_size]

            # Actualiza el mejor individuo global encontrado.
            current_best_in_pop = min(self.population, key=lambda ind: ind.fitness)
            if current_best_in_pop.fitness < best_overall_individual.fitness:
                best_overall_individual = current_best_in_pop.clone()

            if (generation + 1) % (self.num_generations // 10 if self.num_generations >= 10 else 1) == 0:
                print(f"Generation {generation+1}/{self.num_generations}, "
                      f"Best EA Fitness: {best_overall_individual.fitness:.4f}, ")

        return best_overall_individual.bucket_orders, best_overall_individual.weights, best_overall_individual.fitness    


class MAOSBOPSolver:
    """
    Implementa un Algoritmo Memético (MA) para el problema OSBOP.

    Este algoritmo híbrido combina la exploración global de un Algoritmo Evolutivo
    con una fase de explotación local intensiva (aprendizaje individual o "meme")
    para refinar las soluciones y acelerar la convergencia.
    """
    def __init__(self, C_matrix: PrecedenceMatrix, b_num_bucket_orders: int, population_size: int=50, num_generations: int=100, 
                 crossover_prob: float=0.85, mutation_prob_bo: float=0.2, mutation_prob_w: float=0.1,
                 local_search_prob: float=0.5, t_ls_weights: int=100, tournament_size: int = 3, elitism_count: int = 2, ble_max_steps: int = 20, seed: int = None):
        if seed is not None:
            random.seed(seed)

        self.C_matrix = C_matrix
        self.b = b_num_bucket_orders

        # El MA también utiliza la clase base para operaciones comunes de OSBOP.
        self.utility_solver = SLSOSBOPSolver(
            C_matrix=self.C_matrix,
            b_num_bucket_orders=self.b,
            equal_weights_eq=False,  # En MA-OSBOP, los pesos no son iguales.
            t1_outer_iterations=1,  # Valor por defecto, no se usa en MA-OSBOP.
            t2_inner_iterations=t_ls_weights,  # Número de iteraciones internas para cada bucket order.
            seed=seed
        )

        self.population_size = population_size
        self.num_generations = num_generations
        self.p_crossover = crossover_prob
        self.p_mutation_bo = mutation_prob_bo 
        self.p_mutation_w = mutation_prob_w 
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.p_local_search = local_search_prob # Probabilidad de aplicar búsqueda local a un individuo.
        self.ble_max_steps = ble_max_steps  # Número máximo de pasos para la búsqueda local

        self.population: List[Individual] = []
        self.mutation_operators = [
            "bucket_insertion",
            "bucket_interchange",
            "bucket_inversion",
            "bucket_union",
            "bucket_division",
            "item_insertion",
            "item_interchange"
        ]

    def _normalize_weights_inplace(self, weights: WeightVector):
        if not weights: return

        for i in range(len(weights)):
            weights[i] = max(0.01, weights[i])
        
        current_sum = sum(weights)
        if current_sum > 1e-9:
            for i in range(len(weights)):
                weights[i] /= current_sum
        elif len(weights) > 0:  # Si todos los pesos son cero, asignar pesos iguales
            equal_weight = 1.0 / len(weights)
            for i in range(len(weights)):
                weights[i] = equal_weight

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            s_c: SolutionSet = [self.utility_solver._generate_random_bucket_order() for _ in range(self.b)]
            w_c: WeightVector = [1.0 / self.b] * self.b if self.b > 0 else [1.0]
            individual = Individual(s_c, w_c)
            individual.fitness = self.utility_solver._evaluate_solution(individual.bucket_orders, individual.weights)
            self.population.append(individual)

    def _tournament_selection(self) -> Individual:
        tournament = random.sample(self.population, self.tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)
    
    def _crossover_individuals(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        offspring1_bo = [[list(b_item) for b_item in bo] for bo in parent1.bucket_orders]
        offspring1_w = list(parent1.weights)
        offspring2_bo = [[list(b_item) for b_item in bo] for bo in parent2.bucket_orders]
        offspring2_w = list(parent2.weights)

        for k in range(self.b):
            if random.random() < 0.5:
                offspring1_bo[k], offspring2_bo[k] = offspring2_bo[k], offspring1_bo[k]

        if self.b > 0:
            alpha = random.random()
            for k in range(self.b):
                p1_wk = parent1.weights[k]
                p2_wk = parent2.weights[k]
                offspring1_w[k] = alpha * p1_wk + (1 - alpha) * p2_wk
                offspring2_w[k] = (1 - alpha) * p1_wk + alpha * p2_wk

            self._normalize_weights_inplace(offspring1_w)
            self._normalize_weights_inplace(offspring2_w)
        
        return Individual(offspring1_bo, offspring1_w), Individual(offspring2_bo, offspring2_w)
    
    def _mutate_individual(self, individual: Individual):
        for k in range(self.b):
            if random.random() < self.p_mutation_bo:
                chosen_mutation_type = random.choice(self.mutation_operators)
                individual.bucket_orders[k] = self.utility_solver._apply_specific_mutation(individual.bucket_orders[k], chosen_mutation_type)

        if self.b > 0:
            weights_mutated = False
            for k in range(self.b):
                if random.random() < self.p_mutation_w:
                    perturbation = random.gauss(0, 0.1)  # Perturbación gaussiana
                    individual.weights[k] += perturbation
                    weights_mutated = True

            if weights_mutated:
                self._normalize_weights_inplace(individual.weights)

    def _structural_local_search(self, s_c: SolutionSet, w_c: WeightVector) -> SolutionSet:
        """
        Realiza una búsqueda local para mejorar la ESTRUCTURA de los bucket orders.
        """
        current_s_c = copy.deepcopy(s_c)
        current_fitness = self.utility_solver._evaluate_solution(current_s_c, w_c)

        for _ in range(self.ble_max_steps):
            improvement_found = False
            random.shuffle(self.mutation_operators)
            # Explora el vecindario aplicando los operadores de mutación.
            for op in self.mutation_operators:
                neighbor_s_c = copy.deepcopy(current_s_c)
                k_to_mutate = random.randrange(self.b)
                neighbor_s_c[k_to_mutate] = self.utility_solver._apply_specific_mutation(neighbor_s_c[k_to_mutate], op)
                neighbor_fitness = self.utility_solver._evaluate_solution(neighbor_s_c, w_c)

                # Adopta la primera mejora encontrada.
                if neighbor_fitness < current_fitness:
                    current_s_c = neighbor_s_c
                    current_fitness = neighbor_fitness
                    improvement_found = True
                    break # Sale del bucle de operadores y reinicia la búsqueda con la nueva solución.
            
            if not improvement_found:
                break # Si no encuentra mejora en todo el vecindario, se detiene la búsqueda local.

        return current_s_c

    def solve(self) -> Tuple:
        """Ejecuta el algoritmo memético."""
        self._initialize_population()
        best_overall_individual = min(self.population, key=lambda ind: ind.fitness).clone()

        for generation in range(self.num_generations):
            new_population: List[Individual] = []
            if self.elitism_count > 0:
                self.population.sort()
                new_population.extend([ind.clone() for ind in self.population[:self.elitism_count]])

            while len(new_population) < self.population_size:
                p1, p2 = self._tournament_selection(), self._tournament_selection()

                # Cruce y Mutación (igual que en el EA).
                if random.random() < self.p_crossover:
                    offspring1, offspring2 = self._crossover_individuals(p1, p2)
                else:
                    offspring1, offspring2 = p1.clone(), p2.clone()
                
                self._mutate_individual(offspring1)
                self._mutate_individual(offspring2)

                # --- Fase de Aprendizaje Individual (Búsqueda Local) ---
                if random.random() < self.p_local_search:
                     # 1. Mejora la estructura del bucket order.
                    improved_structure_ls1 = self._structural_local_search(offspring1.bucket_orders, offspring1.weights)
                    offspring1.bucket_orders = improved_structure_ls1
                    
                    # 2. Ajusta los pesos para la nueva estructura mejorada.
                    current_fitness = self.utility_solver._evaluate_solution(offspring1.bucket_orders, offspring1.weights)
                    tuned_f1, tuned_w1 = self.utility_solver._tune_weights(
                        s_c=offspring1.bucket_orders, 
                        initial_w_c=offspring1.weights, 
                        initial_f_c=current_fitness
                    )

                    offspring1.weights = tuned_w1
                    offspring1.fitness = tuned_f1
                else:
                    # Si no hay búsqueda local, solo se evalúa.
                    offspring1.fitness = self.utility_solver._evaluate_solution(offspring1.bucket_orders, offspring1.weights)

                new_population.append(offspring1)

                if len(new_population) >= self.population_size:
                    break

                if random.random() < self.p_local_search:
                    improved_structure_ls2 = self._structural_local_search(offspring2.bucket_orders, offspring2.weights)
                    offspring2.bucket_orders = improved_structure_ls2

                    current_fitness = self.utility_solver._evaluate_solution(offspring2.bucket_orders, offspring2.weights)

                    tuned_f2, tuned_w2 = self.utility_solver._tune_weights(
                        s_c=offspring2.bucket_orders, 
                        initial_w_c=offspring2.weights, 
                        initial_f_c=current_fitness
                    )

                    offspring2.weights = tuned_w2
                    offspring2.fitness = tuned_f2
                else:
                    offspring2.fitness = self.utility_solver._evaluate_solution(offspring2.bucket_orders, offspring2.weights)

                
                new_population.append(offspring2)
                if len(new_population) >= self.population_size:
                    break

            self.population = new_population[:self.population_size]    
            current_best_in_pop = min(self.population, key=lambda ind: ind.fitness)
            if current_best_in_pop.fitness < best_overall_individual.fitness:
                best_overall_individual = current_best_in_pop.clone()
            
            if (generation + 1) % (self.num_generations // 10 if self.num_generations >= 10 else 1) == 0:
                print(f"Generation {generation+1}/{self.num_generations}, "
                      f"Best MA Fitness: {best_overall_individual.fitness:.4f}, "
                      f"Best Weights: {[round(w, 3) for w in best_overall_individual.weights]}")
        
        return best_overall_individual.bucket_orders, best_overall_individual.weights, best_overall_individual.fitness
        
    







