# Ausführbare Analyse und historische Notebooks

[01_data_preparation.ipynb](01_data_preparation.ipynb) ist der aktuelle Einstieg. Nach dem dokumentierten Download ruft es ausschließlich `src.data.prepare()` und `load_prepared()` auf. Es zeigt Quelle, Einheiten, Verknüpfung, erhaltene Daten und hypothetische Kappungsauswirkungen mit einem frischen Kernel. Relative Datenpfade beziehen sich auf den Projektstamm.

```powershell
.\.venv\Scripts\python.exe -m src.data --download
.\.venv\Scripts\python.exe scripts/execute_notebook.py notebooks/01_data_preparation.ipynb --output data/notebooks/01_data_preparation.executed.ipynb
```

Das Ausführungswerkzeug verwendet genau den aufrufenden Python-Interpreter und schreibt die ausgeführte Kopie lokal unter `data/notebooks/`. Das versionierte Notebook bleibt ohne gespeicherte Ausgaben. Bei interaktiver Verwendung muss der Kernel dieselbe Projektumgebung verwenden; JupyterLab ist dafür eine optionale Oberfläche.

Die vier ursprünglichen Notebooks liegen bytegetreu in [legacy/](legacy/). Ihre gespeicherten Ergebnisse, alten Datenpfade und Abhängigkeiten dokumentieren die frühere Analyse. Sie gehören nicht zum aktuellen ausführbaren Ablauf und belegen keine korrigierte Modellgüte. Der neue Datenstand ist von ihren Ergebnissen getrennt. Das Ausführungswerkzeug lehnt diese historischen Dateien ausdrücklich ab.

Die aktuelle Vorbereitung enthält keine Modellanpassung und keine Auswahl von Merkmalen, Glättung, Kappungen oder Ausschlüssen. Weitere Analysen folgen erst nach einem festgelegten Validierungsprotokoll. Die vollständige frühere Datenexploration bleibt Teil der dokumentierten Analysegeschichte.
