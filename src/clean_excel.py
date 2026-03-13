import os

datei = "output/predictions.xlsx"  # Hier den Pfad zu deiner Datei eintragen

if os.path.exists(datei):
    os.remove(datei)
    print(f"Datei {datei} wurde gelöscht.")
else:
    print(f"Datei {datei} existiert nicht.")