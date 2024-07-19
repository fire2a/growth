#!python3
"""
simulador de crecimiento usando tabla.csv para modelos de crecimiento con un cambio de manejo

Funciones
* creación de rodal: ha, edad actual, un modelo [Zona, densidad inicial, Índice de Sitio, Manejo, Condición y especie (pino o eucalipto)].
* creación de plan de manejo: raleo, poda, cosecha
* generar biomasa: a partir de un rodal, plan y horizonte; calcular biomasa cambiando de modelo si hay raleo
* generar bosque: para cierto numero de rodales, generar biomasa y plan de manejo, mostrar tablas:
    - BOSQUE: columnas: rodal, plan de manejo, filas: rodal_id_1..n
    - BIOMASA: columnas: biomasa_de_rodal_id_1..n, filas: anos_1..horizonte

ipython
source ~/pyenv/dev/bin/activate
α
β
γ
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml



data=toml.load("input_simulador.toml")
df = pd.read_csv("tabla.csv")
df.index = df.id
# ['id', 'next', 'Especie', 'Zona', 'DensidadInicial', 'SiteIndex', 'Manejo', 'Condicion', 'α', 'β', 'γ']

"""config = {
    'global':{
    "horizonte": 40,
    "rodales": 20,
    "edades": [0, 14],  # falta caso edad < cosecha, luego edad max < cosecha min (14<15)
    "has": [1, 10],
    "raleos": [7, 25],
    "podas": [7, 25],
    "cosechas": [15, 30],
},
'random':{
      "horizonte": 40,
    "rodales": 20,
    "edades": [0, 14],  # falta caso edad < cosecha, luego edad max < cosecha min (14<15)
    "has": [1, 10],
    "raleos": [6, 10],
    "podas": [11, 20],
    "cosechas": [21, 30],
}
}"""


def calc_biomasa(rodal: pd.Series, e: int) -> float:
    """calcular la biomasa para un rodal y una edad, >=0"""
    if e <8:
        return max(e*(rodal["α"] * 9 ** rodal["β"] + rodal["γ"])/9,0)
    else:
        return max(rodal["α"] * e ** rodal["β"] + rodal["γ"], 0)


"""def generar_rodal(idx=None, edades=config["edades"], has=config["has"]) -> pd.Series:
    elegir/sortear un modelo de la tabla, sortear edad y hectareas
    # FIXME : comentar proximas 2 lineas
    # edades=config["edades"]
    # has=config["has"]
    if not idx:
        idx = np.random.choice(df.id)
    return pd.concat(
        (
            df.loc[idx],
            pd.Series({"edad": np.random.randint(*edades), "ha": np.random.randint(*has)}),
        )
    )"""
def generar_rodal(idx=None, edades=data["datos"]["edades"], has=data["datos"]["has"]) -> pd.Series:
    """elegir/sortear un modelo de la tabla, sortear edad y hectareas"""
    # FIXME : comentar proximas 2 lineas
    edades=data["datos"]["edades"]
    has=data["datos"]["has"]
    if not idx:
        idx = np.random.choice(df.id)
    for idx, (edad, ha) in enumerate(zip(edades, has), 1):
        return pd.concat(
            (
                df.loc[idx],  # Datos del DataFrame en el índice específico
                pd.Series({"edad": edad, "ha": ha})  # Edad y hectáreas específicas
            )
        )

def genera_plan_de_manejo(
    raleos=data["usuario"]["raleos"],
    podas=data["usuario"]["podas"],
    cosechas=data["usuario"]["cosechas"],
) -> pd.Series:
    """sortear ano de raleo, poda y cosecha. Debe ser coherente: raleo < cosecha y poda < cosecha"""
    # FIXME : comentar proximas 3 lineas
    raleos=data["usuario"]["raleos"]
    podas=data["usuario"]["podas"]
    cosechas=data["usuario"]["cosechas"]
    #cosecha = np.random.randint(*cosechas)
    for raleo, poda, cosecha in zip(raleos, podas, cosechas):
        # Asegurar la coherencia: raleo < cosecha y poda < cosecha
        if poda >= cosecha:
            poda = cosecha - 1
        if raleo >= cosecha:
            raleo = cosecha - 1
    return pd.Series(
        {
            "raleo": raleos,
            "poda": podas,
            "cosecha": cosechas,
        }
    )

def generar_codigo_kitral(especie: str, edad: int, condición: str) -> str:
    """Genera un diccionario de códigos Kitral basado en la especie, edad y condición"""
    key = (especie, edad, condición)
    if key[0] == "Pinus":
        if key[1] <= 3:
            value = "PL01"
        elif 3 < key[1] <= 11:
            value = "PL05" if key[2] == "con manejo" else "PL02"
        elif 11 < key[1] <= 17:
            value = "PL06" if key[2] == "con manejo" else "PL03"
        else:
            value = "PL07" if key[2] == "con manejo" else "PL04"
    else:  # Eucalyptus
        if key[1] <= 3:
            value = "PL08"
        elif 3 < key[1] <= 10:
            value = "PL09"
        else:
            value = "PL10"
    return value


def simula_rodal_plan(
    rodal=generar_rodal(), plan_mnjo=genera_plan_de_manejo(), horizonte=data["usuario"]["horizonte"]
) -> list[pd.Series]:
    """a partir de un rodal y un plan de manejo, simula la biomasa"""
    # FIXME : comentar proximas 3 lineas
    horizonte = data["usuario"]["horizonte"]
    rodal = generar_rodal()
    plan_mnjo = genera_plan_de_manejo()
    rodal_plan = pd.concat((rodal, plan_mnjo))

    assert plan_mnjo.raleo < plan_mnjo.cosecha < horizonte

    # FIXME : comentar proximas
    # np.isnan(rodal.next)
    if np.isnan(rodal.next):
        rodal_plan.raleo = -1
        periodo_actual= rodal_plan.cosecha-rodal.edad
        
        a = []
        a.append(calc_biomasa(rodal,rodal.edad + i) for i in range(rodal_plan.cosecha-rodal.edad))
        while periodo_actual <horizonte:
            if periodo_actual + rodal_plan.cosecha < horizonte:
                a.append(calc_biomasa(rodal,i) for i in range(rodal_plan.cosecha))
                periodo_actual = periodo_actual + rodal_plan.cosecha
            else:
                a.append(calc_biomasa(rodal,i) for i in range(periodo_actual, horizonte))
                periodo_actual = horizonte
        return rodal.ha * a
            
        """return (
            rodal.ha
            * pd.Series(
                [calc_biomasa(rodal, rodal.edad + i) for i in range(rodal_plan.cosecha-rodal.edad)]
                + [0 for i in range(rodal_plan.cosecha, horizonte)]
            ),
            rodal_plan,
        )"""

    next_rodal = df.loc[rodal.next]
    a=[]
    if rodal.edad >= rodal_plan.raleo:
        if rodal.edad <=10:
            a.append(calc_biomasa(rodal,rodal.edad + i) for i in range(rodal_plan.cosecha-rodal.edad))
        else:
            a.append(calc_biomasa(next_rodal,rodal.edad + i) for i in range(rodal_plan.cosecha-rodal.edad))
    else:
        a.append(calc_biomasa(rodal, rodal.edad + i) for i in range(rodal_plan.raleo-rodal.edad))
        a.append(calc_biomasa(next_rodal, rodal.edad + i) for i in range(rodal_plan.raleo, rodal_plan.cosecha))
    periodo_actual= rodal_plan.cosecha-rodal.edad
    while periodo_actual <horizonte:
            if periodo_actual + rodal_plan.cosecha < horizonte:
                a.append(calc_biomasa(rodal, i) for i in range(rodal_plan.raleo))
                a.append(calc_biomasa(next_rodal, rodal_plan.raleo + i) for i in range(rodal_plan.raleo, rodal_plan.cosecha))
                periodo_actual = periodo_actual + rodal_plan.cosecha
            else:
                if periodo_actual + rodal_plan.raleo < horizonte:
                    a.append(calc_biomasa(rodal,i) for i in range(rodal_plan.raleo))
                    a.append(calc_biomasa(next_rodal,rodal_plan.raleo + i) for i in range(periodo_actual+rodal_plan.raleo,horizonte))
                else:
                    a.append(calc_biomasa(rodal,i) for i in range(periodo_actual, horizonte))
                periodo_actual = horizonte
    return rodal.ha * a

    # print(f"{plan_mnjo.raleo=}", f"{plan_mnjo.cosecha=}", f"{horizonte=}")
    """return (
        rodal.ha
        * pd.Series(
            [calc_biomasa(rodal, rodal.edad + i) for i in range(rodal_plan.raleo-rodal.edad)]
            + [calc_biomasa(next_rodal, rodal.edad + i) for i in range(rodal_plan.raleo, rodal_plan.cosecha)]
            + [0 for i in range(rodal_plan.cosecha, horizonte)]
        ),
        rodal_plan,
    )"""


def simula_tabla(horizonte=data["usuario"]["horizonte"]):
    # FIXME : comentar proxima linea
    horizonte = data["usuario"]["horizonte"]


    # para cada modelo, calcular biomasa hasta horizonte, retorna filas
    df_all = df.apply(lambda row: pd.Series([calc_biomasa(row, e) for e in range(horizonte)]), axis=1)
    # transponer, una columna un modelo
    df_alt = df_all.T

    # graficar
    names = ["Especie", "Zona", "DensidadInicial", "SiteIndex", "Manejo", "Condicion"]
    axes = df_alt.plot()
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.5, box.y0 + box.height])
    # legend_labels = [str(list(df.set_index('id').loc[col][names].to_dict().values())) for col in df_alt.columns]
    legend_labels = [df.loc[col][names].to_dict() for col in df_alt.columns]
    plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig("tabla.png")
    # plt.show()


def simula_bosque(num_rodales=data["datos"]["rodales"]):
    # FIXME : comentar proxima linea
    num_rodales = data["datos"]["rodales"]

    df_bm = pd.DataFrame()
    df_bo = pd.DataFrame()
    for i in range(num_rodales):
        rodal_id = f"rodal_{i}"
        rodal = generar_rodal()
        plan_mnjo = genera_plan_de_manejo()
        biomasa, rodal_plan = simula_rodal_plan(rodal, plan_mnjo)
        # print(f"{rodal_id=}", f"{rodal=}", f"{plan_mnjo=}", f"{biomasa.describe()=}", sep="\n")
        biomasa.name = rodal_id
        rodal_plan.name = rodal_id
        df_bm = pd.concat((df_bm, biomasa), axis=1)
        df_bo = pd.concat(
            (df_bo, rodal_plan.to_frame().T[["id", "next", "edad", "ha", "raleo", "poda", "cosecha"]]), axis=0
        )

    df_bm.to_csv("biomasa1.csv",index=False)
    df_bo.to_csv("bosque1.csv",index=False)

    df_bo.merge(df, on="id")
    df_bos = pd.merge(df_bo, df.drop(["id", "next"], axis=1), how="left", left_on="id", right_on="id")
    df_bos = pd.merge(df_bos, df.drop(["id", "next"], axis=1), how="left", left_on="id", right_on="next")


def main():
    simula_tabla()


if __name__ == "__main__":
    sys.exit(main())

"""
Especie choices ['eucapiltus' 'pino']
Zona choices ['Z1' 'Z2' 'Z6' 'Z7']
DensidadInicial choices [ 800 1250]
SiteIndex choices [24 26 28 30 32 29 23]
Manejo choices [nan 'Intensivo' 'Intensivo2' 'Multipropósito' 'Pulpable']
Condicion choices ['SinManejo' 'PostRaleo1250-700' 'PostPodayRaleo700-300']
α choices [2.37100e+00 5.46400e+00 7.70900e+00 1.02280e+01 2.37900e+00 3.79400e+00
 5.30700e+00 7.77900e+00 1.03040e+01 3.22100e+00 3.61200e+00 4.55900e+00
 5.92600e+00 7.30000e+00 3.19700e+00 3.72900e+00 5.12900e+00 6.20100e+00
 7.59200e+00 4.21000e-01 5.76000e-01 1.22491e+02 2.29340e+01 7.00000e-02
 4.09290e+01 6.90000e-02 4.70000e-02 5.83780e+01 5.68160e+01 9.59200e+00
 2.08520e+01]
β choices [1.778 1.591 1.519 1.462 1.774 1.666 1.6   1.513 1.457 1.647 1.656 1.618
 1.567 1.532 1.65  1.645 1.574 1.547 1.514 2.253 2.044 0.562 0.936 2.85
 0.774 2.791 2.905 0.701 0.666 1.169 0.844]
γ choices [   0.       6.24     5.66     3.485    3.968 -416.814 -171.991    7.522
 -227.526    1.441    8.454 -298.45  -260.576  -97.246 -103.893]
"""
