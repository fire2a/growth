#!/usr/bin/env python
"""
Generador de rodales con distintos planes de manejo, crecimiento de acuerdo a modelos de crecimiento (tabla.csv)

Uso:
    - editar config.toml
    - ejecutar en consola: python simulator.py
    - ver: python -c "import simulator; simulator.generate()"
    - guardar: python -c "import simulator; rodales = simulator.generate(); simulator.write(rodales)"

Interactive use:
    $ ipython
    In [1]: from simulator import *
    In [2]: rodales = simulator.generate()

Funciones principales:
    - calc_biomass: calcular la biomasa para un model y una edad
    - print_manejos_possibles: imprimir los manejos posibles
    - write: escribir archivos de salida
    - generate: generar rodales con distintos planes de manejo

Funciones auxiliares:
    - plot_models: graficar modelos de crecimiento
    - solve_numeric: resolver numericamente los zeros de cada ecuación de crecimiento
    - solve_symbolic: resolver simbolicamente la ecuación de crecimiento, calcular los zeros
    - append_zeros: agregar ceros a los modelos

Variables globales:
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


def calc_biomass(model: np.void, e: int) -> float:
    """calcular la biomasa para un model y una edad
    Si la edad es menor que el zero del model, se redondea hacia arriba la edad donde es cero y
    Se pondera linealmente
    """
    if e < model["zero"]:
        e_up = np.ceil(model["zero"])
        return max(
            e / e_up * (model["α"] * e_up ** model["β"] + model["γ"]),
            0,  # demuestre que siempre es positivo
        )
    else:
        return max(model["α"] * e ** model["β"] + model["γ"], 0)


def generar_codigo_kitral(especie: str, edad: int, condicion: str) -> str:
    """Genera un diccionario de códigos Kitral basado en la Especie, edad y condición"""
    if especie == "pino":
        if edad <= 3:
            value = 19
        elif 3 < edad <= 11:
            value = 23 if condicion == "con manejo" else 20
        elif 11 < edad <= 17:
            value = 24 if condicion == "con manejo" else 21
        else:
            value = 25 if condicion == "con manejo" else 22
    elif especie == "eucalyptus":  # Eucalyptus
        if edad <= 3:
            value = 26
        elif 3 < edad <= 10:
            value = 27
        else:
            value = 28
    else:
        print("error, especie desconocida")
        return -9999  # mala practica cambiar el tipo de retorno
    return value


def print_manejos_possibles():
    """imprime todos los manejos posibles para los rodales"""
    manejos_posibles = []
    print("manejos posibles", end=": ")
    for cosecha, raleo in product(np.arange(*config["pino"]["cosechas"]), np.arange(*config["pino"]["raleos"])):
        if raleo > cosecha:
            continue
        print(f"(c{cosecha}, r{raleo})", end=", ")
        manejos_posibles.append([int(raleo), int(cosecha)])
    for cosecha in np.arange(*config["eucalyptus"]["cosechas"]):
        print(f"(c{int(cosecha)}, r{-1})", end=", ")
        manejos_posibles.append([-1, int(cosecha)])
    for cosecha in np.arange(*config["pino"]["cosechas"]):
        print(f"(c{cosecha}, r{-1})", end=", ")
        manejos_posibles.append([-1, int(cosecha)])

    for raleo in np.arange(*config["pino"]["raleos"]):
        print(f"(c{-1}, r{raleo})", end=", ")
        manejos_posibles.append([int(raleo), -1])

    print()
    return manejos_posibles


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


def generate():
    """Genera los rodales con las biomasas generadas por cada año, dependiendo de su manejo y edad de crecimiento, junto con la biomasa para vender y el codigo kitral"""
    rodales = []
    exclude_ids = [22, 23, 26, 27, 30, 31]
    # Filtrar para excluir los modelos con ids en exclude_ids
    filtered_models = [m for m in models if m["id"] not in exclude_ids]
    # itera = iter(range(config["rodales"]))
    # r = next(itera)
    for r in range(config["rodales"]):
        model = rng.choice(filtered_models)
        # model = rng.choice(models)
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
            # "edades": edades, -> adentro de manejos
            "ha": ha,
        }
        rodales += [rodal]
        display(rodal)
        manejos = []

        # has cosecha if any of the proposed "cosechas" ranges are in the simulated "edades"
        has_cosecha = any(np.isin(np.arange(*config[model["Especie"]]["cosechas"]), edades))

        # can have raleo only if it's pino
        if model["Especie"] == "pino":
            if not has_cosecha:
                # easy case
                has_raleo = (model["next"] != -1) and any(np.isin(np.arange(*config["pino"]["raleos"]), edades))
            else:
                # adjust "edades" -> "edades_manejo", by periodically "cosecha" (to harvest) via modulus operator
                # then check if raleo is in range
                for cosecha in np.arange(*config["pino"]["cosechas"]):
                    edades_manejo = edades % cosecha
                    has_raleo = (model["next"] != -1) and any(
                        np.isin(np.arange(*config["pino"]["raleos"]), edades_manejo)
                    )
                    print(cosecha, np.arange(*config["pino"]["raleos"]), edades_manejo, has_raleo)
                    if has_raleo:
                        break
                has_raleo = False
        else:
            has_raleo = False

        print(f"{r=}, {has_cosecha=}, {has_raleo=}")
        # 4 cases combinations of "has_cosecha" and "has_raleo"
        if not has_cosecha and not has_raleo:
            manejo = {
                "rid": r,
                "cosecha": -1,
                "raleo": -1,
                "biomass": ha * np.array([calc_biomass(model, e) for e in edades]),
                "edades": edades,
                "eventos": ["" for e in edades],
                "vendible": [0 for e in edades],
                "codigo_kitral": [generar_codigo_kitral(model["Especie"], e, "sin manejo") for e in edades],
            }
            manejos = [manejo]
            display(manejo)
        elif has_cosecha and not has_raleo:
            manejos = []
            # iterb = iter(np.arange(*config[model["Especie"]]["cosechas"]))
            # cosecha = next(iterb)
            for cosecha in np.arange(*config[model["Especie"]]["cosechas"]):
                if cosecha not in edades:  # edades_manejo mejor que edades ?
                    display(f"skipping: {e0=} !< {cosecha=} !< {e1=}")
                    continue
                edades_manejo = edades % cosecha
                manejo = {
                    "rid": r,
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
            manejos = []
            # iterc = iter(np.arange(*config[model["Especie"]]["raleos"]))
            # raleo = next(iterc)
            for raleo in np.arange(*config[model["Especie"]]["raleos"]):
                if raleo not in edades:
                    display(f"skipping: {e0=} !< {raleo=} !< {e1=}")
                    continue
                manejo = {
                    "rid": r,
                    "cosecha": -1,
                    "raleo": raleo,
                    "biomass": ha
                    * np.array(
                        [
                            calc_biomass(model, e) if e <= raleo else calc_biomass(models[model["next"]], e)
                            for e in edades
                        ]
                    ),
                    "edades": edades,
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
        else:  # has cosecha and raleo, se asume que se raleo antes del periodo 0 en calc_biomass
            manejos = []
            # iterd = iter(
            #     product(
            #         np.arange(*config[model["Especie"]]["cosechas"]), np.arange(*config[model["Especie"]]["raleos"])
            #     )
            # )
            # cosecha, raleo = next(iterd)
            for cosecha, raleo in product(
                np.arange(*config[model["Especie"]]["cosechas"]), np.arange(*config[model["Especie"]]["raleos"])
            ):
                edades_manejo = edades % cosecha
                if (
                    (raleo >= cosecha) or (cosecha not in edades) or (raleo not in edades_manejo)
                ):  # edades_manejo mejor que edades ?
                    display(f"skipping: {min(edades_manejo)=} !< {raleo=} !< {cosecha=} !< {e1=}")
                    continue
                mods = [model["id"] if e < raleo else model["next"] for e in edades_manejo]
                display(f"{mods=}")
                eventos = []
                vendible = []
                for e in edades_manejo:
                    if e == raleo:
                        eventos += ["r"]
                        vendible += [(calc_biomass(model, raleo) - calc_biomass(models[model["next"]], raleo))]
                    elif e == 0:
                        eventos += ["c"]
                        vendible += [calc_biomass(models[model["next"]], cosecha)]
                    else:
                        eventos += [""]
                        vendible += [0]
                manejo = {
                    "rid": r,
                    "cosecha": cosecha,
                    "raleo": raleo,
                    "biomass": ha
                    * np.array([0 if e == 0 else calc_biomass(models[m], e) for m, e in zip(mods, edades_manejo)]),
                    "edades": edades_manejo,
                    "eventos": eventos,
                    "vendible": ha * np.array(vendible),
                    "codigo_kitral": [
                        (
                            generar_codigo_kitral(model["Especie"], e, "con manejo")
                            if e >= raleo
                            else (
                                generar_codigo_kitral(
                                    model["Especie"], cosecha, "con manejo"
                                )  # FIJO en cosecha ENTRE 1..raleo ?
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
        display(manejos)
    display(rodales)
    return rodales


def main():
    print(__doc__)
    rodales = simulator.generate()
    simulator.write(rodales)


if __name__ == "__main__":
    main()
