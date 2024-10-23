# GROWTH SIMULATOR

A timber plantation growth simulator, with stands and management policies options.

Based on !["Modelos de predicción de biomasa a nivel de rodal en plantaciones de Eucalyptus globulus y Pinus radiata en Zona centro sur de en Chile"]('Modelos%20de%20predicción%20de%20biomasa%20a%20nivel%20de%20rodal%20en%20plantaciones%20de%20Eucalyptus%20globulus%20en%20Chile.pdf'). By: Alejandro Miranda, Blas Mola and Víctor Hinojosa

1. For 34 types of eucalyptus and pinus plantations in the central-south of Chile, a statistical study fitted the following power law (models.csv):
   
$$
biomass(t) = \alpha \cdot t^\beta + \gamma
$$

2. A template for generating a timber plantation and different management policies was made (config.toml)
   ```toml
   horizonte = 10 # number of years to generate
   rodales = 36 # number of stands, choosing one model at random

   [random]
   seed = 4 # random number generator seed
   edades = [1, 18] # min, max age of generated stands
   has = [5, 15] #  min, max hectares of generated stands

   [pino]
   raleos = [6, 11, 1] # for each Pinus stand, generate different biomass history considering thinnig in the year 6, 7, ... 11.
   cosechas = [18, 29, 4] # for each stand, generate different biomass history considering harvesting in the year 18, 22, 16 (every 4 years) 

   [eucalyptus]
   cosechas = [10, 20, 2] # for each Eucalyptus stand, generate different biomass history considering harvesting in the year 10, 12, 14, ... 20 (every 2 years) 
   ```



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
