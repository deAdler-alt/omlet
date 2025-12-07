import numpy as np
from src.propagation_model import PropagationModel
from src.energy_model import EnergyModel

class WBANFitness:
    def __init__(self, config, weights, scenario_id='S1'):
        self.config = config
        self.weights = weights # {'w_E': 0.5, 'w_R': 0.5}
        self.scenario = config['scenarios'][scenario_id]
        self.sensors_def = self.scenario['sensors']
        
        self.prop_model = PropagationModel(config)
        self.energy_model = EnergyModel(config)
        
        # Maksymalna moc nadajnika zdefiniowana w scenariuszu
        self.max_p_tx = self.scenario['P_TX_max']
    
    def get_metrics(self, solution):
        """
        Pomocnicza metoda zwracająca szczegółowe metryki dla danego rozwiązania.
        Używana po optymalizacji do raportowania wyników (nie wpływa na fitness).
        """
        decoded = self.decode_solution(solution)
        sensors = decoded[:-1]
        hub = decoded[-1]
        
        total_energy_J = 0.0
        reliability_penalty = 0.0
        geometric_penalty = 0.0
        node_energies = [] # Do obliczenia czasu życia
        
        # 1. Walidacja geometryczna
        for s in sensors:
            zone_name = s['def']['zone']
            zone_limits = self.config['body_zones'][zone_name]
            x, y = s['pos']
            if not (zone_limits['x_min'] <= x <= zone_limits['x_max'] and 
                    zone_limits['y_min'] <= y <= zone_limits['y_max']):
                geometric_penalty += 1000.0

        # 2. Fizyka
        for s in sensors:
            d = self.prop_model.calculate_distance(s['pos'], hub['pos'])
            is_los = self.prop_model.is_line_of_sight(s['pos'], hub['pos'])
            pl_db = self.prop_model.calculate_path_loss(d, is_los)
            req_tx = self.energy_model.calculate_required_tx_power(pl_db)
            
            margin = self.max_p_tx - req_tx
            if margin < 0:
                reliability_penalty += abs(margin) * 50.0
                req_tx = self.max_p_tx
            
            bits_per_sec = s['def']['data_rate']
            energy_node = self.energy_model.calculate_energy_consumption(bits_per_sec, req_tx)
            
            total_energy_J += energy_node
            node_energies.append(energy_node)
            
        # 3. Obliczenie czasu życia sieci (First Node Dies)
        # T_life = E_init / max(E_node)  (liczba sekund/rund)
        e_init = self.config['energy_model']['E_init']
        if max(node_energies) > 0:
            network_lifetime = e_init / max(node_energies)
        else:
            network_lifetime = 0

        return {
            'E_total_real': total_energy_J,
            'P_rel': reliability_penalty,
            'P_geo': geometric_penalty,
            'T_life': network_lifetime
        }


    def decode_solution(self, solution):
        """
        Dekoduje płaski wektor rozwiązania na listę współrzędnych.
        solution: [x1, y1, x2, y2, ..., xN, yN, x_hub, y_hub]
        """
        coords = []
        num_sensors = len(self.sensors_def)
        
        # Sensory
        for i in range(num_sensors):
            coords.append({
                'type': 'sensor',
                'id': self.sensors_def[i]['id'],
                'pos': (solution[2*i], solution[2*i+1]),
                'def': self.sensors_def[i]
            })
            
        # Hub (ostatnie 2 wartości)
        hub_idx = 2 * num_sensors
        coords.append({
            'type': 'hub',
            'pos': (solution[hub_idx], solution[hub_idx+1])
        })
        
        return coords

    def evaluate(self, solution):
        """
        Główna funkcja celu F(g).
        Zwraca wartość do minimalizacji.
        """
        decoded = self.decode_solution(solution)
        sensors = decoded[:-1]
        hub = decoded[-1]
        
        total_energy = 0.0
        reliability_penalty = 0.0
        geometric_penalty = 0.0
        
        # 1. Kara geometryczna (czy sensory są w swoich strefach?)
        for s in sensors:
            zone_name = s['def']['zone']
            zone_limits = self.config['body_zones'][zone_name]
            x, y = s['pos']
            
            # Sprawdź granice
            if not (zone_limits['x_min'] <= x <= zone_limits['x_max'] and 
                    zone_limits['y_min'] <= y <= zone_limits['y_max']):
                geometric_penalty += 1000.0 # Duża kara za wyjście poza strefę
                
        # Walidacja Huba (Hub musi być na torsie w S1/S2)
        # (Można to zmienić w zależności od założeń, tutaj zakładamy strefę 'chest' lub 'waist')
        hx, hy = hub['pos']
        # Przykładowo hub może być gdziekolwiek na ciele, ale nie poza nim.
        # Dla uproszczenia nie karzemy huba, chyba że wyjdzie poza [0,1].
        
        if geometric_penalty > 0:
            return 1e6 + geometric_penalty # Early exit dla złych rozwiązań

        # 2. Obliczenia Energii i Niezawodności
        for s in sensors:
            # Odległość
            d = self.prop_model.calculate_distance(s['pos'], hub['pos'])
            
            # LOS/NLOS
            is_los = self.prop_model.is_line_of_sight(s['pos'], hub['pos'])
            
            # Tłumienie
            pl_db = self.prop_model.calculate_path_loss(d, is_los)
            
            # Wymagana moc
            req_tx = self.energy_model.calculate_required_tx_power(pl_db)
            
            # Czy mieścimy się w budżecie mocy? (Niezawodność)
            margin = self.max_p_tx - req_tx
            if margin < 0:
                # Kara za brak zasięgu (wymagana moc > dostępna moc)
                reliability_penalty += abs(margin) * 50.0
                # Ustawiamy moc na max dostępną do obliczeń energii (chociaż transmisja się nie uda)
                req_tx = self.max_p_tx
            
            # Energia (dla 1 pakietu o wielkości np. 1000 bitów * data_rate)
            # Upraszczamy: energia na sekundę transmisji
            bits_per_sec = s['def']['data_rate']
            energy_node = self.energy_model.calculate_energy_consumption(bits_per_sec, req_tx)
            
            total_energy += energy_node

        # Agregacja (Funkcja celu 2.6)
        # Normalizacja składników jest ważna! Energia jest rzędu 1e-6, kary rzędu 10-100.
        # Musimy przeskalować energię, żeby była widoczna dla algorytmu.
        
        # Skalowanie energii do rzędu jedności (np. mnożymy przez 1e6 uJ)
        normalized_energy = total_energy * 1e4 
        
        fitness = (self.weights['w_E'] * normalized_energy + 
                   self.weights['w_R'] * reliability_penalty + 
                   geometric_penalty)
                   
        return fitness
