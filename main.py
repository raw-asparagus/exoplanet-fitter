import os

from astropy.time import Time
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from rvfitter import *


def main():
    DATA_DIR = os.path.join(".", "data")

    keck_csv = os.path.join(DATA_DIR, "marcy-2001-keck.csv")
    keck_df = pd.read_csv(keck_csv)
    keck_df["JD"] = keck_df["JD-2450000"] + 2.45e6

    lick_csv = os.path.join(DATA_DIR, "marcy-2001-lick.csv")
    lick_df = pd.read_csv(lick_csv)
    lick_df["JD"] = keck_df["JD-2450000"] + 2.45e6

    # Dataset to use
    df = keck_df

    p1: np.longdouble = np.longdouble(60.85)
    t_peri1: np.longdouble = np.longdouble(2450301)
    k1: np.longdouble = np.longdouble(239)
    e1: np.longdouble = np.longdouble(0.27)
    omega1: np.longdouble = np.longdouble(np.radians(24))
    v0: np.longdouble = np.longdouble(0.0)
    marcy_2001: NDArray[np.longdouble] = np.array([
        p1, t_peri1, k1, e1, omega1,
        v0
    ])

    fitter = Fitter(df["JD"].values, df["relative velocity"].values,
                    df["uncertainty"].values)
    fitter.fit(marcy_2001)

    ts: NDArray[np.longdouble] = (
        np.linspace(df["JD"].min(), df["JD"].max(), 1000, dtype=np.longdouble))
    computed_rvs: NDArray[np.longdouble] = fitter.model.evaluate(ts)

    fig = plt.figure(figsize=(10, 6), dpi=300)
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Time(ts, format="jd").plot_date, computed_rvs,
             label='Fitted RV')
    ax1.errorbar(Time(df["JD"], format="jd").plot_date,
                 df["relative velocity"], yerr=df["uncertainty"],
                 label='Relative velocity', linestyle='', marker='.')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    ax1.set(xlabel='Date [$\\text{yr}$]', ylim=(-500, 500),
            ylabel=r'Relative velocity [$\text{ms}^{'r'-1}$]',
            title="Radial velocity fit to Marcy 2001 (Keck)")
    ax1.legend()
    ax1.grid(True, alpha=0.5, linestyle='--')

    plt.savefig(os.path.join(".", "plots", "marcy-2001-rv-fit.svg"))

    for i, planet in enumerate(fitter.model.planets):
        base = i * 5
        period_unc = fitter.uncertainties[base]
        t_peri_unc = fitter.uncertainties[base + 1]
        k_unc = fitter.uncertainties[base + 2]
        e_unc = fitter.uncertainties[base + 3]
        omega_unc = fitter.uncertainties[base + 4]

        print(f"Planet {i + 1}:")
        print(f"\tPeriod: {planet.period:.4f} days ± {period_unc:.4f}")
        print(f"\tTime of periastron: {planet.t_peri:.4f} ± {t_peri_unc:.4f}")
        print(f"\tSemi-amplitude (K): {planet.k:.4f} m/s ± {k_unc:.4f}")
        print(f"\tEccentricity (e): {planet.e:.4f} ± {e_unc:.4f}")
        print(
            f"\tArgument of periastron (omega): {np.degrees(planet.omega):.4f} ± "
            f"{np.degrees(omega_unc):.4f}")


if __name__ == "__main__":
    main()
