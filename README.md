# GROWTH SIMULATOR

A timber plantation growth simulator, with stands and management policies options.

Based on !["Modelos de predicción de biomasa a nivel de rodal en plantaciones de Eucalyptus globulus y Pinus radiata en Zona centro sur de en Chile"](Modelos%20de%20predicción%20de%20biomasa%20a%20nivel%20de%20rodal%20en%20plantaciones%20de%20Eucalyptus%20globulus%20en%20Chile.pdf). By: Alejandro Miranda, Blas Mola and Víctor Hinojosa

1. For 34 types of eucalyptus and pinus plantations in the central-south of Chile, a statistical study fitted the following power law (tabla.csv):
   
$$
biomass(t) = \alpha \cdot t^\beta + \gamma
$$

2. For some types of pinus, in the initial years of the simulation, the formula for biomass(t) was yielding negative values. Therefore, the formula used in the simulator is:

$$
\text{biomass}(t) = 
\begin{cases} 
\left(\alpha \cdot \text{stableyear}^\beta + \gamma \right) \cdot \frac{t}{\text{stableyear}} & \text{if } t < \text{stableyear} \\
\alpha \cdot t^\beta + \gamma & \text{if} t >= \text{stableyear}
\end{cases}
$$
Stable year is the year when the formula begins to yield stable results (it depends on the type of pinus).

3. A template for generating a timber plantation and different management policies was made (config.toml)
    ```toml
    horizonte = 10 # number of years to generate
    rodales = 36 # number of stands, choosing one model at random

    [random]
    # random number generator seed: omit or comment for different results each run
    seed = 5
    # `low` (inclusive) to `high` (exclusive)
    # n, n+1 for getting a single value: n
    # see np.random.randint
    edades = [1, 18] # min, max age of generated stands
    has = [5, 15] #  min, max hectares of generated stands

    # ranges: start, stop, step
    # n, n+1 for getting a single value range: [n]
    # see np.arange
    [eucalyptus]
    cosechas = [10, 20, 3] # for each Eucalyptus stand, generate different biomass histories harvesting in the year 10, 13, 16 and 19 (4 histories) 
    
    [pino]
    raleos = [6, 11, 2] # thinning policies in the years 6, 8, 10.
    cosechas = [18, 29, 3] # harvesting policies in the years 18, 21, 24, ... (every 3 years)
    # all feasible histories combining thinning and harvesting policies will be generated
    min_ral = 6 #minimun age where you can thinn a tree
    ```
2.  If you have a real forest you want to simulate, you will need a `.csv` or `.shp` file containing the forest data. This data should include the following fields:

- **id**: Unique identifier for each stand  
- **age**: Age of the stand  
- **hectare (ha)**: Area in hectares  
- **fid**: File ID of the `.csv` or `.shp` file  

You can then create a `bosque_data.csv` file using the auxiliary functions `get_data` and `create_bosque`.

### quick start

0. Clone, download or just get `simulator.py`, `tabla.csv`, `bosque_data.csv` and `config.toml`
1. Have numpy installed (and toml if python version < 3.11)
2. Console run: `python simulator.py` to generate csv files or `python simulator.py -r` if you don't have the csv data and you want to create a random forest (or you can create a bosque_data.csv with get_data() and create_bosque() from auxiliary)
3. Scripting/Interactive use:
    ```python
    import simulator
    rodales = simulator.main(['-s'])
    ```

### more info

1. `python simulator.py --help`
2. See `example` folder
3. Read ![`simulator.py`](simulator.py) `__doc__`s
4. Read `config.toml`
5. Read Methodoly section of !["Modelos de predicción de biomasa a nivel de rodal en plantaciones de Eucalyptus globulus y Pinus radiata en Zona centro sur de en Chile"](Modelos%20de%20predicción%20de%20biomasa%20a%20nivel%20de%20rodal%20en%20plantaciones%20de%20Eucalyptus%20globulus%20en%20Chile.pdf)

### models plots

![models](models.png)
![tabla](tabla.png)
### example of new formula of biomass in model whit id 30
![1_id](model_30_id.png)  
