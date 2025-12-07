import sys
import os
import yaml
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Dodajemy katalog główny do ścieżki, aby importować moduły src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fitness_function import WBANFitness

# Import algorytmów i przestrzeni z mealpy (v3.0+)
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO
from mealpy.utils.space import FloatVar

def load_config(path=None):
    """
    Bezpieczne ładowanie konfiguracji niezależnie od miejsca uruchomienia skryptu.
    """
    if path is None:
        # Pobierz ścieżkę do folderu, w którym jest ten skrypt (experiments/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Wyjdź piętro wyżej (do omlet/) i wejdź do config/
        project_root = os.path.dirname(current_dir)
        path = os.path.join(project_root, 'config', 'wban_params.yaml')

    print(f"📂 Ładowanie konfiguracji z: {path}")
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def run_experiment_batch():
    config = load_config()
    
    # Definicja parametrów eksperymentu
    n_runs = 50       # Liczba powtórzeń
    epochs = 100      # Liczba iteracji
    pop_size = 50     # Rozmiar populacji
    
    # Warianty wag (Energia vs Niezawodność)
    weight_variants = {
        'Balanced': {'w_E': 0.5, 'w_R': 0.5},
        'EnergyFirst': {'w_E': 0.8, 'w_R': 0.2},
        'ReliabilityFirst': {'w_E': 0.2, 'w_R': 0.8}
    }
    
    # Algorytmy do przetestowania
    algorithms = {
        'GA': GA.BaseGA, 
        'PSO': PSO.OriginalPSO
    }

    # Scenariusze z pliku config
    scenarios = config['scenarios'].keys()
    
    # Utworzenie folderu na wyniki
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'raw', datetime.now().strftime('%Y%m%d_%H%M'))
    results_dir = os.path.abspath(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"🚀 Rozpoczynam symulację WBAN...")
    print(f"📂 Wyniki trafią do: {results_dir}")
    
    for scenario_id in scenarios:
        print(f"\n🔹 Scenariusz: {scenario_id}")
        
        # Pobierz liczbę sensorów dla scenariusza
        n_sensors = len(config['scenarios'][scenario_id]['sensors'])
        # Wymiar problemu: 2 * n_sensors (x,y dla każdego sensora) + 2 (x,y dla huba)
        problem_dim = 2 * n_sensors + 2
        
        # Definicja granic (bounds) dla Mealpy v3+
        # Używamy FloatVar, ponieważ współrzędne są ciągłe [0.0, 1.0]
        lb = [0.0] * problem_dim
        ub = [1.0] * problem_dim
        bounds = FloatVar(lb=lb, ub=ub, name="wban_positions")
        
        for w_name, weights in weight_variants.items():
            print(f"  ⚖️ Wagi: {w_name} {weights}")
            
            # Inicjalizacja funkcji celu
            fitness_obj = WBANFitness(config, weights, scenario_id)
            
            for algo_name, AlgoClass in algorithms.items():
                print(f"    🤖 Algorytm: {algo_name}", end=" ")
                
                run_data = []
                
                for run in range(n_runs):
                    print(".", end="", flush=True)
                    
                    # Słownik problemu zgodny z Mealpy v3+
                    problem_dict = {
                        "obj_func": fitness_obj.evaluate, # Funkcja celu
                        "bounds": bounds,                 # Granice zmiennych
                        "minmax": "min",                  # Minimalizacja
                        "log_to": None,                   # Wyłączenie logowania
                    }
                    
                    # Inicjalizacja modelu
                    model = AlgoClass(epoch=epochs, pop_size=pop_size)
                    
                    start_time = time.time()
                    
                    # --- POPRAWKA TUTAJ ---
                    # model.solve() zwraca obiekt Agent, z którego wyciągamy solution i fitness
                    best_agent = model.solve(problem_dict)
                    
                    best_position = best_agent.solution
                    best_fitness = best_agent.target.fitness
                    # ----------------------

                    exec_time = time.time() - start_time
                    
                    # Zbieranie szczegółowych metryk (rozkład na energię, kary itp.)
                    decoded_metrics = fitness_obj.get_metrics(best_position)
                    
                    # Historia zbieżności
                    convergence = model.history.list_global_best_fit
                    
                    record = {
                        'Scenario': scenario_id,
                        'Weights': w_name,
                        'Algorithm': algo_name,
                        'Run': run,
                        'Best_Fitness': best_fitness,
                        'Execution_Time': exec_time,
                        'Energy_Total_J': decoded_metrics['E_total_real'],
                        'Reliability_Penalty': decoded_metrics['P_rel'],
                        'Geometric_Penalty': decoded_metrics['P_geo'],
                        'Network_Lifetime_Rounds': decoded_metrics['T_life'],
                        'Convergence': str(convergence), # Zapis jako string, żeby nie rozwaliło CSV
                        'Best_Solution_Vector': str(list(best_position))
                    }
                    run_data.append(record)
                
                # Zapisz wyniki
                df = pd.DataFrame(run_data)
                filename = f"{scenario_id}_{w_name}_{algo_name}.csv"
                df.to_csv(os.path.join(results_dir, filename), index=False)
                print(" Done.")

if __name__ == "__main__":
    run_experiment_batch()