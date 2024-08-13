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
import sys
from itertools import product

import numpy as np

np.set_printoptions(precision=1)

if "IPython" in sys.modules:
    from IPython.display import display
else:
    # Here be dragons
    display = print

if sys.version_info >= (3, 11):
    import tomllib

    with open("config_test.toml", "rb") as f:
        config = tomllib.load(f)
else:
    import toml

    config = toml.load("config.toml")
display(f"{config=}")


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


def plot_models(horizon: int = 40, show=True, save=False):
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
    """resolver numericamente los zeros de cada ecuación de crecimiento"""
    from scipy.optimize import fsolve

    zeros = []
    for model in models:
        zeros += [fsolve(lambda x: model["α"] * x ** model["β"] + model["γ"], config["horizonte"])[0]]
    print(f"numeric_{zeros=}")
    return zeros


def solve_symbolic():
    """calcula el zeros simbolico de la ecuacion de biomasa"""
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
    """agregar zeros a los modelos"""
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


def generar_codigo_kitral(especie: str, edad: int, condicion: str) -> str:
    """Genera un diccionario de códigos Kitral basado en la Especie, edad y condición"""
    if especie == "pino":
        if edad <= 3:
            value = "PL01"
        elif 3 < edad <= 11:
            value = "PL05" if condicion == "con manejo" else "PL02"
        elif 11 < edad <= 17:
            value = "PL06" if condicion == "con manejo" else "PL03"
        else:
            value = "PL07" if condicion == "con manejo" else "PL04"
    elif especie == "eucalyptus":  # Eucalyptus
        if edad <= 3:
            value = "PL08"
        elif 3 < edad <= 10:
            value = "PL09"
        else:
            value = "PL10"
    else:
        print("error, especie desconocida")
        return
    return value


def print_manejos_possibles():
    """imprime todos los manejos posibles para los rodales"""
    print("manejos posibles", end=": ")
    for cosecha, raleo in product(np.arange(*config["random"]["cosechas_p"]), np.arange(*config["random"]["raleos"])):
        if raleo > cosecha:
            continue
        print(f"(c{cosecha}, r{raleo})", end=", ")
    for cosecha in np.arange(*config["random"]["cosechas_e"]):
        print(f"(c{cosecha}, r{-1})", end=", ")
    print()


def write(rodales):
    """Crea los csv de salida, con la biomasa, eventos, biomasa vendible y codigos de kitral"""
    bm = np.array([manejo["biomass"] for rodal in rodales for manejo in rodal["manejos"]])
    ev = np.array([manejo["eventos"] for rodal in rodales for manejo in rodal["manejos"]])
    vd = np.array([manejo["vendible"] for rodal in rodales for manejo in rodal["manejos"]])
    kt = np.array([manejo["codigo_kitral"] for rodal in rodales for manejo in rodal["manejos"]])
    names = ",".join(
        [f"R{rodal['rid']}_c{manejo['cosecha']}_r{manejo['raleo']}" for rodal in rodales for manejo in rodal["manejos"]]
    )
    names = names.replace("_r-1", "").replace("_c-1", "")
    np.savetxt("biomass.csv", bm.T, delimiter=",", header=names, comments="")
    np.savetxt("events.csv", ev.T, delimiter=",", header=names, comments="", fmt="%s")
    np.savetxt("vendible.csv", vd.T, delimiter=",", header=names, comments="")
    np.savetxt("codigo_kitral.csv", kt.T, delimiter=",", header=names, comments="", fmt="%s")

    bos_names = ["rid", "mid", "edad_inicial", "ha"]  # aprender hacer formato decente
    bos = np.array([tuple(r[k] for k in bos_names) for r in rodales])
    np.savetxt("bosque.csv", bos, delimiter=",", header=",".join(bos_names), comments="", fmt="%d")


def generate2():
    """Genera los rodales con las biomasas generadas por cada año, dependiendo de su manejo y edad de crecimiento, junto con la biomasa para vender y el codigo kitral"""
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
                "vendible": [0 for e in edades],
                "codigo_kitral": [generar_codigo_kitral(model["Especie"], e, "sin manejo") for e in edades],
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
                    "vendible": ha * np.array([calc_biomass(model, cosecha) if e == 0 else 0 for e in edades_manejo]),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], cosecha, "sin manejo")
                            if e == 0
                            else generar_codigo_kitral(model["Especie"], e, "sin manejo")
                        )
                        for e in edades_manejo
                    ],
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
                    "vendible": ha
                    * np.array(
                        [
                            (
                                (calc_biomass(model, raleo) - calc_biomass(models[model["next"]], raleo))
                                if e == raleo
                                else 0
                            )
                            for e in edades
                        ]
                    ),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], e, "sin manejo")
                            if e < raleo
                            else generar_codigo_kitral(model["Especie"], e, "con manejo")
                        )
                        for e in edades
                    ],
                }
                manejos += [manejo]
                display(manejo)
        else:
            for cosecha, raleo in product(
                np.arange(*config["random"]["cosechas"]), np.arange(*config["random"]["raleos"])
            ):
                edades_manejo = edades % cosecha
                if (raleo >= cosecha) or (cosecha not in edades) or (raleo not in edades_manejo):
                    print(f"skipping: {min(edades_manejo)=} !< {raleo=} !< {cosecha=} !< {e1=}")
                    continue
                mods = [model["id"] if e <= raleo else model["next"] for e in edades_manejo]
                print(f"{mods=}")
                eventos = []
                vendible = []
                for e in edades_manejo:
                    if e == raleo:
                        eventos += ["r"]
                        vendible += [(calc_biomass(model, raleo) - calc_biomass(models[model["next"]], raleo))]
                        # [f"m{mods[e-1]}-r->m{mods[e]}" for e in edades_manejo] e-1 in edades ?
                    elif e == 0:
                        eventos += ["c"]
                        vendible += [calc_biomass(models[model["next"]], cosecha)]
                    else:
                        eventos += [""]
                        vendible += [0]
                manejo = {
                    "cosecha": cosecha,
                    "raleo": raleo,
                    "biomass": ha * np.array([calc_biomass(models[m], e) for m, e in zip(mods, edades_manejo)]),
                    "edades": edades_manejo,
                    "eventos": eventos,
                    "vendible": ha * np.array(vendible),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], e, "con manejo")
                            if e >= raleo
                            else (
                                generar_codigo_kitral(model["Especie"], cosecha, "con manejo")
                                if e == 0
                                else generar_codigo_kitral(model["Especie"], e, "sin manejo")
                            )
                        )
                        for e in edades_manejo
                    ],
                }
                manejos += [manejo]
                display(manejo)
        rodal["manejos"] = manejos
        # display(manejos)
    # display(rodales)
    return rodales


def generate():
    """Genera los rodales con las biomasas generadas por cada año, dependiendo de su manejo y edad de crecimiento, junto con la biomasa para vender y el codigo kitral"""
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
        if model["Especie"] == "pino":
            has_cosecha = any(np.isin(np.arange(*config["random"]["cosechas_p"]), edades))
        else:
            has_cosecha = any(np.isin(np.arange(*config["random"]["cosechas_e"]), edades))
        if not has_cosecha:
            has_raleo = (model["next"] != -1) and any(np.isin(np.arange(*config["random"]["raleos"]), edades))
        else:  # has_cosecha
            for cosecha, raleo in product(
                np.arange(*config["random"]["cosechas_p"]), np.arange(*config["random"]["raleos"])
            ):
                edades_manejo = edades % cosecha
                has_raleo = (model["next"] != -1) and any(
                    np.isin(np.arange(*config["random"]["raleos"]), edades_manejo)
                )

                # Si hay raleo, detener la búsqueda
                if has_raleo:
                    break

        # print(f"{r=}, {has_cosecha=}, {has_raleo=}")
        if not has_cosecha and not has_raleo:
            manejo = {
                "cosecha": -1,
                "raleo": -1,
                "biomass": ha * np.array([calc_biomass(model, e) for e in edades]),
                "eventos": ["" for e in edades],
                "vendible": [0 for e in edades],
                "codigo_kitral": [generar_codigo_kitral(model["Especie"], e, "sin manejo") for e in edades],
            }
            manejos += [manejo]
            display(manejo)
        elif has_cosecha and not has_raleo:
            if model["Especie"] == "pino":
                cos = config["random"]["cosechas_p"]
            else:
                cos = config["random"]["cosechas_e"]
            for cosecha in np.arange(*cos):
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
                    "vendible": ha * np.array([calc_biomass(model, cosecha) if e == 0 else 0 for e in edades_manejo]),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], cosecha, "sin manejo")
                            if e == 0
                            else generar_codigo_kitral(model["Especie"], e, "sin manejo")
                        )
                        for e in edades_manejo
                    ],
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
                    "vendible": ha
                    * np.array(
                        [
                            (
                                (calc_biomass(model, raleo) - calc_biomass(models[model["next"]], raleo))
                                if e == raleo
                                else 0
                            )
                            for e in edades
                        ]
                    ),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], e, "sin manejo")
                            if e < raleo
                            else generar_codigo_kitral(model["Especie"], e, "con manejo")
                        )
                        for e in edades
                    ],
                }
                manejos += [manejo]
                display(manejo)
        else:
            for cosecha, raleo in product(
                np.arange(*config["random"]["cosechas_p"]), np.arange(*config["random"]["raleos"])
            ):
                edades_manejo = edades % cosecha
                if (raleo >= cosecha) or (cosecha not in edades) or (raleo not in edades_manejo):
                    print(f"skipping: {min(edades_manejo)=} !< {raleo=} !< {cosecha=} !< {e1=}")
                    continue
                mods = [model["id"] if e <= raleo else model["next"] for e in edades_manejo]
                print(f"{mods=}")
                eventos = []
                vendible = []
                for e in edades_manejo:
                    if e == raleo:
                        eventos += ["r"]
                        vendible += [(calc_biomass(model, raleo) - calc_biomass(models[model["next"]], raleo))]
                        # [f"m{mods[e-1]}-r->m{mods[e]}" for e in edades_manejo] e-1 in edades ?
                    elif e == 0:
                        eventos += ["c"]
                        vendible += [calc_biomass(models[model["next"]], cosecha)]
                    else:
                        eventos += [""]
                        vendible += [0]
                manejo = {
                    "cosecha": cosecha,
                    "raleo": raleo,
                    "biomass": ha * np.array([calc_biomass(models[m], e) for m, e in zip(mods, edades_manejo)]),
                    "edades": edades_manejo,
                    "eventos": eventos,
                    "vendible": ha * np.array(vendible),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], e, "con manejo")
                            if e >= raleo
                            else (
                                generar_codigo_kitral(model["Especie"], cosecha, "con manejo")
                                if e == 0
                                else generar_codigo_kitral(model["Especie"], e, "sin manejo")
                            )
                        )
                        for e in edades_manejo
                    ],
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
