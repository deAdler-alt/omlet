import numpy as np
import math

class PropagationModel:
    def __init__(self, config):
        """
        Inicjalizacja modelu propagacji na podstawie pliku config.
        Parametry pochodzą z [19] Deepak & Babu.
        """
        self.params = config['propagation_model']
        self.zones = config['body_zones']

    def calculate_distance(self, p1, p2):
        """Odległość euklidesowa w metrach (zakładając, że mapa 1x1 to np. 1.8m x 0.5m w rzeczywistości).
        Dla uproszczenia przyjmujemy skalowanie 1 jednostka = 1 metr."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_line_of_sight(self, sensor_pos, hub_pos):
        """
        Wykrywa czy połączenie jest LOS (Line-of-Sight) czy NLOS (Non-Line-of-Sight).
        Heurystyka geometryczna na podstawie [10] Januszkiewicz:
        Jeśli linia prosta między sensorem a hubem przecina strefę 'chest' (tors),
        a żaden z punktów nie leży w 'chest', następuje cieniowanie (NLOS).
        """
        # Jeśli oba punkty są w tej samej strefie -> LOS
        # (Uproszczenie: zakłada brak przeszkód wewnątrz jednej strefy)
        
        # Pobierz granice torsu
        chest = self.zones['chest']
        
        # Sprawdź przecięcie odcinka z prostokątem torsu
        # (Implementacja algorytmu Cohena-Sutherlanda lub uproszczona)
        if self._line_intersects_rect(sensor_pos, hub_pos, chest):
            # Jeśli jeden z punktów jest wewnątrz torsu, to nie jest przeszkoda, to miejsce montażu.
            # NLOS występuje tylko gdy transmisja przebiega "przez" ciało (np. plecy -> przód).
            # Tutaj upraszczamy: jeśli linia przecina tors, a sensor jest np. na plecach (nie modelujemy 3D),
            # traktujemy to jako potencjalny NLOS dla bezpieczeństwa.
            return False # NLOS
            
        return True # LOS

    def calculate_path_loss(self, distance, is_los):
        """
        Oblicza tłumienie ścieżki (Path Loss) w dB.
        Wzór (1.2) z pracy dyplomowej / Eq. (1) z [19].
        """
        # Zabezpieczenie przed d=0
        d = max(distance, 0.01)
        d0 = self.params['d0']

        if is_los:
            pl0 = self.params['PL0_LOS']
            n = self.params['n_LOS']
            sigma = self.params['sigma_LOS']
        else:
            pl0 = self.params['PL0_NLOS']
            n = self.params['n_NLOS'] # Tutaj n jest znacznie wyższe (5.9)
            sigma = self.params['sigma_NLOS']

        # Składnik losowy (Shadowing) - normalizacja
        shadowing = np.random.normal(0, sigma)
        
        # Formuła IEEE 802.15.6
        pl = pl0 + 10 * n * np.log10(d / d0) + shadowing
        
        return pl

    def _line_intersects_rect(self, p1, p2, rect):
        """Pomocnicza funkcja: czy odcinek p1-p2 przecina prostokąt zdefiniowany w rect."""
        # Prostokąt definiują (min_x, min_y) i (max_x, max_y)
        # To jest uproszczona wersja, dla pełnej precyzji w 2D
        # sprawdzamy przecięcie z każdą z 4 krawędzi.
        
        # Szybki test bounding box
        return not (max(p1[0], p2[0]) < rect['x_min'] or 
                    min(p1[0], p2[0]) > rect['x_max'] or 
                    max(p1[1], p2[1]) < rect['y_min'] or 
                    min(p1[1], p2[1]) > rect['y_max'])
