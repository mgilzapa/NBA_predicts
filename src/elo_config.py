"""elo_config.py — gemeinsame ELO-Konstanten.

Wird von feature_engineering.py (ELO-Berechnung) und season_simulator.py
(Saison-Prognose) importiert, damit die Werte nicht auseinanderlaufen.
"""

ELO_START = 1500
ELO_K = 20
ELO_K_PRESEASON = 8          # Preseason: Stars spielen wenig → schwaecheres Signal
ELO_SCALE = 400              # Skala der ELO-Erwartungsformel
HCA = 100                    # Heimvorteil in ELO-Punkten (+100 ≈ 3-4 Punkte NBA-Heimvorteil)
