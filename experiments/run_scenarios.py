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
# Import algorytmów z mealpy
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import PSO

def load_config(path='config/wban_params.yaml'):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def run_experiment_batch():
    config = load_config()
    
    # Definicja parametrów eksperymentu
    n_runs = 50  # Liczba powtórzeń dla statystyki
    epochs = 100 # Liczba iteracji algorytmu
    pop_size = 50 # Rozmiar populacji
    
    # Warianty wag (Energia vs Niezawodność)
    weight_variants = {
        'Balanced': {'w_E': 0.5, 'w_R': 0.5},
        'EnergyFirst': {'w_E': 0.8, 'w_R': 0.2},
        'ReliabilityFirst': {'w_E': 0.2, 'w_R': 0.8}
    }
    
    # Algorytmy do przetestowania (klasy z Mealpy)
    algorithms = {
        'GA': GA.BaseGA, 
        'PSO': PSO.OriginalPSO
    }

    # Scenariusze z pliku config
    scenarios = config['scenarios'].keys() # np. ['S1']
    
    results_dir = f"results/raw/{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Rozpoczynam symulację WBAN...")
    print(f"Wyniki trafią do: {results_dir}")
    
    for scenario_id in scenarios:
        print(f"\n Scenariusz: {scenario_id}")
        
        # Pobierz liczbę sensorów dla scenariusza, aby ustalić wymiar problemu
        n_sensors = len(config['scenarios'][scenario_id]['sensors'])
        # Wymiar D = 2 * n_sensors (x,y dla każdego) + 2 (x,y huba)
        problem_dim = 2 * n_sensors + 2
        
        # Granice przestrzeni poszukiwań [0, 1] dla każdej zmiennej
        lb = [0.0] * problem_dim
        ub = [1.0] * problem_dim
        
        for w_name, weights in weight_variants.items():
            print(f"Wagi: {w_name} {weights}")
            
            # Inicjalizacja funkcji celu dla danej konfiguracji
            fitness_obj = WBANFitness(config, weights, scenario_id)
            
            for algo_name, AlgoClass in algorithms.items():
                print(f"Algorytm: {algo_name}", end=" ")
                
                run_data = []
                
                for run in range(n_runs):
                    print(".", end="", flush=True)
                    
                    # Definicja problemu dla Mealpy
                    # fit_func: funkcja którą minimalizujemy
                    # minmax: "min" dla minimalizacji
                    problem_dict = {
                        "fit_func": fitness_obj.evaluate,
                        "lb": lb,
                        "ub": ub,
                        "minmax": "min",
                    }
                    
                    # Inicjalizacja i uruchomienie modelu
                    # verbose=False wyłącza logi konsolowe mealpy
                    model = AlgoClass(epoch=epochs, pop_size=pop_size)
                    
                    start_time = time.time()
                    best_position, best_fitness = model.solve(problem_dict)
                    exec_time = time.time() - start_time
                    
                    # Zbieranie szczegółowych metryk dla najlepszego rozwiązania
                    # Musimy je ponownie policzyć, aby rozbić na składowe (Energia, Kary)
                    # ponieważ solve() zwraca tylko skalarną wartość fitness
                    # (Można to zoptymalizować, ale dla analizy jest bezpieczniej tak)
                    
                    # Dekodujemy rozwiązanie, aby pobrać parametry fizyczne
                    decoded_metrics = fitness_obj.get_metrics(best_position)
                    
                    # Zapis historii zbieżności (convergence)
                    # model.history.list_global_best_fit zawiera fitness w każdej epoce
                    convergence = model.history.list_global_best_fit
                    
                    record = {
                        'Scenario': scenario_id,
                        'Weights': w_name,
                        'Algorithm': algo_name,
                        'Run': run,
                        'Best_Fitness': best_fitness,
                        'Execution_Time': exec_time,
                        'Energy_Total_J': decoded_metrics['E_total_real'], # Prawdziwa energia w Dżulach (bez normalizacji)
                        'Reliability_Penalty': decoded_metrics['P_rel'],
                        'Geometric_Penalty': decoded_metrics['P_geo'],
                        'Network_Lifetime_Rounds': decoded_metrics['T_life'],
                        'Convergence': convergence, # Zapisujemy jako listę/string
                        'Best_Solution_Vector': list(best_position)
                    }
                    run_data.append(record)
                
                # Zapisz wyniki dla danej kombinacji do CSV
                df = pd.DataFrame(run_data)
                filename = f"{scenario_id}_{w_name}_{algo_name}.csv"
                df.to_csv(os.path.join(results_dir, filename), index=False)
                print(" Done.")

if __name__ == "__main__":
    run_experiment_batch()