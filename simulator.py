#!/usr/bin/env ipython
"""
Generador de rodales con distintos planes de manejo, crecimiento de acuerdo a modelos de crecimiento (tabla.csv)

Uso:
    - editar config.toml
    - ver: python -c "import simulator; simulator.generate()"
    - guardar: python -c "import simulator; rodales = simulator.generate(); simulator.write(rodales)"

Interactive use:
    $ ipython
    In [1]: from simulator import *

Otras funciones:
    - plot_models: graficar modelos de crecimiento
    - solve_numeric: resolver numericamente los zeros de cada ecuación de crecimiento
    - solve_symbolic: resolver simbolicamente la ecuación de crecimiento, calcular los zeros
    - append_zeros: agregar ceros a los modelos
    - calc_biomass: calcular la biomasa para un model y una edad
    - print_manejos_possibles: imprimir los manejos posibles
    - write: escribir archivos de salida
    - generate: generar rodales con distintos planes de manejo

GLOBALS:
    - config: configuración leida de config.toml
    - models: modelos de crecimiento leidos de tabla.csv
    - rng: generador de números aleatorios: ojo con la semilla
"""
import toml
from itertools import product

import numpy as np
from IPython.display import display
#display = print
np.set_printoptions(precision=1)

#with open("config.toml", "rb") as f:
#    config = toml.load(f)
#    display(f"{config=}")
config=toml.load("config.toml")


# reproducible
rng = np.random.default_rng(config["random"]["seed"])
# random
# rng = np.random.default_rng()

models = np.genfromtxt(
    "tabla.csv",
    delimiter=",",
    names=True,  # id,next,Especie,Zona,DensidadInicial,SiteIndex,Manejo,Condicion,α,β,γ
    dtype=None,
    encoding="utf-8",
)
# models.dtype.names
# model.dtype.names
# dict(zip(model.dtype.names,model))

# id index from 0
# models[ models['id'] == num ] == models[num]

# models with raleo
# models[ models['next'] !=-1 ]['id']
# OJO -1 is assigned by default


def plot_models(horizon:int=40 , show=True, save=False):
    """Crea geafico con los calculos de biomasa por cada id del arbol eje x igual al año y eje y igual a la biomasa"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots() 
    ax.set_title("Modelos de crecimiento")
    for model in models:
        x = np.linspace(0, horizon, (horizon - 0) * 2)
        y = model["α"] * x ** model["β"] + model["γ"]
        ax.plot(x, y, label=model["id"])
    ax.axhline(0, color="black", linestyle="--")
    ax.legend()
    if show:
        plt.show()
    if save:
        plt.savefig("models.png")


def solve_numeric():
    from scipy.optimize import fsolve  

    zeros = []
    for model in models:
        zeros += [fsolve(lambda x: model["α"] * x ** model["β"] + model["γ"], config["horizonte"])[0]]
    print(f"numeric_{zeros=}")
    return zeros


def solve_symbolic():
    from sympy import solve, symbols

    x, α, β, γ = symbols("x α β γ")
    print(solve(α * x**β + γ, x))
    # [(-γ/α)**(1/β)]
    zeros = []
    for model in models:
        zeros += [(-model["γ"] / model["α"]) ** (1 / model["β"])]
    print(f"symbolic_{zeros=}")
    return zeros


def append_zeros():
    global models
    import numpy.lib.recfunctions as rfn

    for name, zeros in zip(["zero_numeric", "zero_symbolic"], [solve_numeric(), solve_symbolic()]):
        models = rfn.append_fields(models, name, zeros, usemask=False)

    zn = models["zero_numeric"]
    zs = models["zero_symbolic"]
    mask = ~np.isnan(zn) & ~np.isnan(zs)
    assert np.isclose(zn[mask], zs[mask]).all()

    names = ",".join(models.dtype.names)
    np.savetxt("new_table.csv", models, delimiter=",", header=names, comments="", fmt="%s")


def calc_biomass(model: np.void, e: int) -> float:
    """calcular la biomasa para un model y una edad"""
    return max(model["α"] * e ** model["β"] + model["γ"], 0)


def print_manejos_possibles():
    """imprime todos los posibles manejos para los rodales"""
    print("manejos posibles", end=": ")
    for cosecha, raleo in product(np.arange(*config["random"]["cosechas"]), np.arange(*config["random"]["raleos"])):
        if raleo > cosecha:
            continue
        print(f"(c{cosecha}, r{raleo})", end=", ")
    print()


def write(rodales):
    """Crea los csv con los datos de las biomasas calcula"""
    bm = np.array([manejo["biomass"] for rodal in rodales for manejo in rodal["manejos"]])
    ev = np.array([manejo["eventos"] for rodal in rodales for manejo in rodal["manejos"]])
    names = ",".join(
        [f"R{rodal['rid']}_c{manejo['cosecha']}_r{manejo['raleo']}" for rodal in rodales for manejo in rodal["manejos"]]
    )
    names = names.replace("_r-1", "").replace("_c-1", "")
    np.savetxt("biomass.csv", bm.T, delimiter=",", header=names, comments="")
    np.savetxt("events.csv", ev.T, delimiter=",", header=names, comments="", fmt="%s")

    bos_names = ["rid", "mid", "edad_inicial", "ha"] #aprender hacer formato decente 
    bos = np.array([tuple(r[k] for k in bos_names) for r in rodales])
    np.savetxt("bosque.csv", bos, delimiter=",", header=",".join(bos_names), comments="")


def generate():
    """Genera los rodales con las biomasas generadas por cada año, dependiendo de su manejo y edad de crecimiento """
    rodales = []
    for r in range(config["rodales"]):
        model = rng.choice(models)
        # print(model)
        e0 = rng.integers(*config["random"]["edades"])
        e1 = e0 + config["horizonte"]
        edades = np.arange(e0, e1)
        ha = rng.integers(*config["random"]["has"])
        rodal = {
            "rid": r,
            "mid": model["id"],
            "edad_inicial": e0,
            "edad_final": e1,
            "edades": edades,
            "ha": ha,
        }
        rodales += [rodal]
        display(rodal)
        manejos = []
        has_cosecha = any(np.isin(np.arange(*config["random"]["cosechas"]), edades))
        has_raleo = (model["next"] != -1) and any(np.isin(np.arange(*config["random"]["raleos"]), edades))
        # print(f"{r=}, {has_cosecha=}, {has_raleo=}")
        if not has_cosecha and not has_raleo:
            manejo = {
                "cosecha": -1,
                "raleo": -1,
                "biomass": ha * np.array([calc_biomass(model, e) for e in edades]),
                "eventos": ["" for e in edades],
            }
            manejos += [manejo]
            display(manejo)
        elif has_cosecha and not has_raleo:
            for cosecha in np.arange(*config["random"]["cosechas"]):
                if cosecha not in edades:
                    print(f"skipping: {e0=} !< {cosecha=} !< {e1=}")
                    continue
                edades_manejo = edades % cosecha 
                manejo = {
                    "cosecha": cosecha,
                    "raleo": -1,
                    "biomass": ha * np.array([calc_biomass(model, e) for e in edades_manejo]),
                    "edades": edades_manejo,
                    "eventos": ["c" if e == 0 else "" for e in edades_manejo],
                }
                manejos += [manejo]
                display(manejo)
        elif not has_cosecha and has_raleo:
            for raleo in np.arange(*config["random"]["raleos"]):
                if raleo not in edades:
                    print(f"skipping: {e0=} !< {raleo=} !< {e1=}")
                    continue
                manejo = {
                    "cosecha": -1,
                    "raleo": raleo,
                    "biomass": ha
                    * np.array(
                        [
                            calc_biomass(model, e) if e <= raleo else calc_biomass(models[model["next"]], e)
                            for e in edades
                        ]
                    ),
                    "eventos": ["r" if e == raleo else "" for e in edades],
                }
                manejos += [manejo]
                display(manejo)
        else:
            for cosecha, raleo in product(
                np.arange(*config["random"]["cosechas"]), np.arange(*config["random"]["raleos"])
            ):
                if (raleo >= cosecha) or (cosecha not in edades) or (raleo not in edades):
                    print(f"skipping: {e0=} !< {raleo=} !< {cosecha=} !< {e1=}")
                    continue
                edades_manejo = edades % cosecha
                mods = [model["id"] if e <= raleo else model["next"] for e in edades_manejo]
                print(f"{mods=}")
                eventos = []
                for e in edades_manejo:
                    if e == raleo:
                        eventos += ["r"]
                        # [f"m{mods[e-1]}-r->m{mods[e]}" for e in edades_manejo] e-1 in edades ?
                    elif e == 0:
                        eventos += ["c"]
                    else:
                        eventos += [""]
                manejo = {
                    "cosecha": cosecha,
                    "raleo": raleo,
                    "biomass": ha * np.array([calc_biomass(models[m], e) for m, e in zip(mods, edades_manejo)]),
                    "edades": edades_manejo,
                    "eventos": eventos,
                }
                manejos += [manejo]
                display(manejo)
        rodal["manejos"] = manejos
        # display(manejos)
    # display(rodales)
    return rodales


def superpro():
    import pickle

    # store
    with open("all.pickle", "wb") as f:
        pickle.dump((config, models, rodales), f)

    # pickup
    with open("all.pickle", "rb") as f:
        config, models, rodales = pickle.load(f)


def main():
    print(__doc__)
    print_manejos_possibles()


if __name__ == "__main__":
    main()
