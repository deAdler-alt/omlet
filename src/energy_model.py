class EnergyModel:
    def __init__(self, config):
        """
        Inicjalizacja modelu energetycznego.
        Parametry z [19] Deepak & Babu (2016).
        """
        self.params = config['energy_model']
        self.prop_params = config['propagation_model']

    def calculate_required_tx_power(self, path_loss_db):
        """
        Oblicza wymaganą moc nadawania (w dBm), aby sygnał dotarł do odbiornika
        z wymaganym marginesem bezpieczeństwa.
        Wzór (2.3) z pracy.
        """
        p_sens = self.prop_params['P_sens']
        m_safe = self.prop_params['M_safe']
        
        # Bilans łącza: P_TX >= P_sens + PL + Margin
        required_p_tx_dbm = p_sens + path_loss_db + m_safe
        return required_p_tx_dbm

    def calculate_energy_consumption(self, bits, tx_power_dbm):
        """
        Oblicza energię (J) zużytą na przesłanie 'bits' bitów przy mocy nadawania 'tx_power_dbm'.
        Model radia pierwszego rzędu.
        """
        # Konwersja dBm na Waty
        tx_power_watts = 10 ** ((tx_power_dbm - 30) / 10)
        
        # Pobór energii przez elektronikę (stały)
        e_elec = self.params['E_elec'] * bits
        
        # Pobór energii przez wzmacniacz (zależny od mocy nadawania)
        # Przybliżenie: czas trwania bitu * moc * sprawność
        # Zakładamy np. bitrate 1Mbps dla uproszczenia czasu trwania, 
        # lub że energia wzmacniacza to P_TX * t_bit.
        # W modelu uproszczonym z [20] E_amp jest stałe na bit/m^2,
        # ale tutaj mamy lepszy model zależny od P_TX.
        
        # E_total = E_elec_TX + E_amp
        # E_amp ≈ (P_TX / BitRate) (upraszczając)
        # Tutaj użyjemy modelu liniowego z Deepaka:
        
        # Całkowita energia na bit przy danej mocy P_TX:
        # E_bit = E_elec + (P_TX / R_bit) ?
        # Dla celów optymalizacji przyjmiemy, że koszt energetyczny wzmacniacza 
        # skaluje się z wymaganą mocą wypromieniowaną.
        
        e_amp = tx_power_watts * bits * 1e-6 # Skalowanie (uproszczone dla symulacji)
        
        total_energy = e_elec + e_amp
        return total_energy
