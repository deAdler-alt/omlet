import pandas as pd
import os
import glob

def aggregate_results(results_root_dir='results/raw'):
    """
    Szuka najnowszego folderu z wynikami i łączy wszystkie pliki CSV.
    """
    # Znajdź najnowszy folder
    subdirs = [os.path.join(results_root_dir, d) for d in os.listdir(results_root_dir)]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    
    print(f"Agregacja wyników z: {latest_subdir}")
    
    all_files = glob.glob(os.path.join(latest_subdir, "*.csv"))
    df_list = []
    
    for filename in all_files:
        df = pd.read_csv(filename)
        df_list.append(df)
        
    if not df_list:
        print("Brak plików CSV do połączenia.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Oblicz statystyki zbiorcze (średnia, std) dla każdej konfiguracji
    summary = full_df.groupby(['Scenario', 'Weights', 'Algorithm']).agg({
        'Best_Fitness': ['mean', 'std', 'min'],
        'Energy_Total_J': ['mean', 'std'],
        'Network_Lifetime_Rounds': ['mean', 'std'],
        'Execution_Time': ['mean']
    }).reset_index()
    
    # Spłaszcz nazwy kolumn
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    output_path = os.path.join(latest_subdir, 'final_summary.xlsx')
    summary.to_excel(output_path, index=False)
    
    print(f"Raport zbiorczy zapisano w: {output_path}")
    print("\n--- Podgląd wyników (Top 5) ---")
    print(summary.head())

if __name__ == "__main__":
    aggregate_results()