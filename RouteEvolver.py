import sys
import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

"""
RouteEvolver: Genetic Algorithm for TSP Optimization.
Author: Pablo Flores
Description: Solves the Traveling Salesman Problem using evolutionary strategies
(Tournament Selection, Two-Point Crossover, Swap Mutation).
"""

class City:
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y

    def __repr__(self):
        return f"City {self.id} ({self.x}, {self.y})"

# --- Configuration ---
GENERATIONS = 150
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3

# --- Core Functions ---

def plot_and_save_solution(cities: List[City], solution: List[int], generation: int):
    """Generates a visual representation of the current best path."""
    # Extract coordinates based on solution order
    x_coords = [cities[i].x for i in solution] + [cities[solution[0]].x]
    y_coords = [cities[i].y for i in solution] + [cities[solution[0]].y]
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='#2563EB', label='Route', linewidth=1.5)
    
    # Annotate cities
    for city in cities:
        plt.text(city.x, city.y, str(city.id), fontsize=9, ha='right', color='#DC2626', weight='bold')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Route Evolution - Generation {generation}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    filename = f"generation_{generation}.png"
    plt.savefig(filename, dpi=100)
    plt.close()

def calculate_distance(city1: City, city2: City) -> float:
    return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)

def fitness(sol: List[int], cities: List[City]) -> float:
    """Calculates total distance of the path (lower is better)."""
    total_dist = 0
    num_cities = len(sol)
    for i in range(num_cities):
        ind1 = sol[i]
        ind2 = sol[(i+1) % num_cities]
        total_dist += calculate_distance(cities[ind1], cities[ind2])
    return total_dist

def tournament_selection(population: List[List[int]], fitness_map: Dict[Tuple[int], float]) -> List[List[int]]:
    """Selects the fittest individuals using tournament strategy."""
    selected = []
    for _ in range(len(population)):
        participants = random.sample(population, TOURNAMENT_SIZE)
        best = min(participants, key=lambda ind: fitness_map[tuple(ind)])
        selected.append(best)
    return selected

def two_points_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Combines two parents to produce offspring using two-point crossover."""
    n = len(parent1)
    l, r = sorted(random.sample(range(n), 2))
    
    # Initialize children with placeholders
    child1 = [None] * n
    child2 = [None] * n
    
    # Copy the segment between l and r
    child1[l:r+1] = parent1[l:r+1]
    child2[l:r+1] = parent2[l:r+1]
    
    # Fill the remaining positions
    fill_child(child1, parent2, r, n)
    fill_child(child2, parent1, r, n)
    
    return child1, child2

def fill_child(child: List[int], parent: List[int], r: int, n: int):
    """Helper to fill remaining genes maintaining order."""
    current_pos = (r + 1) % n
    for gene in parent:
        if gene not in child:
            # Find next empty spot
            while child[current_pos] is not None:
                current_pos = (current_pos + 1) % n
            child[current_pos] = gene

def apply_mutation(population: List[List[int]]) -> List[List[int]]:
    """Randomly swaps cities in a route based on mutation rate."""
    new_population = []
    for individual in population:
        if random.random() < MUTATION_RATE:
            ind_copy = individual[:]
            idx1, idx2 = random.sample(range(len(ind_copy)), 2)
            ind_copy[idx1], ind_copy[idx2] = ind_copy[idx2], ind_copy[idx1]
            new_population.append(ind_copy)
        else:
            new_population.append(individual)
    return new_population

# --- Main Execution ---

if __name__ == "__main__":
    # Raw Data Input
    input_text = """
    1 832.78 48.15
    2 -765.88 -708.65
    3 469.01 586.08
    4 602.09 -149.98
    5 -627.21 -482.60
    6 -401.81 -199.65
    7 503.02 945.53
    8 207.66 -493.33
    9 935.36 641.91
    10 -326.89 -736.58
    11 -891.14 258.82
    12 703.66 -257.23
    13 -534.94 -531.85
    14 -650.01 705.85
    15 982.88 -71.73
    16 -103.29 -853.56
    17 664.37 -61.78
    18 524.06 618.21
    19 -273.99 311.45
    20 -111.50 -493.26
    21 414.50 -149.26
    22 -143.08 796.02
    23 -170.75 430.26
    24 844.32 723.33
    """

    # Parse Data
    cities = []
    for line in input_text.strip().split("\n"):
        parts = line.split()
        cities.append(City(int(parts[0]), float(parts[1]), float(parts[2])))

    n_cities = len(cities)
    
    # Initialize Population
    population = [random.sample(range(n_cities), n_cities) for _ in range(POPULATION_SIZE)]

    print(f"Starting Genetic Evolution over {GENERATIONS} generations...")

    for gen in range(GENERATIONS):
        # Calculate fitness for caching
        fitness_map = {tuple(ind): fitness(ind, cities) for ind in population}
        
        # Log progress
        best_sol_gen = min(population, key=lambda ind: fitness_map[tuple(ind)])
        best_dist = fitness_map[tuple(best_sol_gen)]
        
        if gen % 10 == 0 or gen == GENERATIONS - 1:
            print(f"Generation {gen}: Best Distance = {best_dist:.2f}")
            plot_and_save_solution(cities, best_sol_gen, gen)

        # Evolution Steps
        selection = tournament_selection(population, fitness_map)
        
        # Crossover
        next_gen = []
        num_cross = int(len(selection) * CROSSOVER_RATE // 2)
        for _ in range(num_cross):
            p1, p2 = random.sample(selection, 2)
            c1, c2 = two_points_crossover(p1, p2)
            next_gen.extend([c1, c2])
        
        # Fill rest with elites/random
        while len(next_gen) < POPULATION_SIZE:
            next_gen.append(random.choice(selection))
            
        population = apply_mutation(next_gen)

    print("Optimization Complete. Check generation images.")