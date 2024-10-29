#!/usr/bin/env python
import numpy as np

models = np.genfromtxt(
    "tabla.csv",
    delimiter=",",
    names=True,  # id,next,Especie,Zona,DensidadInicial,SiteIndex,Manejo,Condicion,α,β,γ
    dtype=None,
    encoding="utf-8",
)


def plot_models(horizon: int = 40, show=True, save=False):
    """Crea grafico con los calculos de biomasa por cada id del arbol eje x igual al año y eje y igual a la biomasa"""
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
    display(f"numeric_{zeros=}")
    return zeros


def solve_symbolic():
    """calcula el zeros simbolico de la ecuacion de biomasa"""
    from sympy import solve, symbols

    x, α, β, γ = symbols("x α β γ")
    display(solve(α * x**β + γ, x))
    # [(-γ/α)**(1/β)]
    zeros = []
    for model in models:
        zeros += [(-model["γ"] / model["α"]) ** (1 / model["β"])]
    display(f"symbolic_{zeros=}")
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


def superpro():
    import pickle

    # store
    with open("all.pickle", "wb") as f:
        pickle.dump((config, models, rodales), f)

    # pickup
    with open("all.pickle", "rb") as f:
        config, models, rodales = pickle.load(f)
