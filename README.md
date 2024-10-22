# GROWTH SIMULATOR

A timber plantation growth simulator, with stands and management policies options.

Based on !["Modelos de predicción de biomasa a nivel de rodal en plantaciones de Eucalyptus globulus y Pinus radiata en Zona centro sur de en Chile"]('Modelos%20de%20predicción%20de%20biomasa%20a%20nivel%20de%20rodal%20en%20plantaciones%20de%20Eucalyptus%20globulus%20en%20Chile.pdf'). By: Alejandro Miranda, Blas Mola and Víctor Hinojosa

1. For 34 types of eucalyptus and pinus plantations in the central-south of Chile, a statistical study fitted the following power law:
$$
biomass(t) = \alpha \cdot t^\beta + \gamma
$$
These 34 stand types and parameters are stored in models.csv

2. 


### quick start

1. text edit `config.toml` for the desired parameters
2. run: `python -c "import simulator; rodales = simulator.generate(); simulator.write(rodales)"`

### more options

see `simulator.py`

### requirements

numpy

#### optionals

matplotlib
scipy
sympy

![models](models.png)
![tabla](tabla.png)
