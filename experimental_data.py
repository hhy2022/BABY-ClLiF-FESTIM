"""BABY-1L run data: schedules, irradiations, measured releases and the tritium source.

Measurements are LIBRA's published processed_data.json for each run (cached in data/,
downloaded from the pinned URLs below when missing).  Times are days from the first
irradiation of the run.
"""

import functools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
DAY = 86400.0
BQ_PER_ATOM = np.log(2.0) / (12.32 * 365.25 * DAY)  # tritium decay constant [1/s]

RUN_URLS = {
    1: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-1/refs/tags/v0.6/data/processed_data.json",
    2: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-2/refs/tags/v0.5/data/processed_data.json",
    3: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-3/refs/tags/v0.2/data/processed_data.json",
    4: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-4/refs/tags/v0.1/data/processed_data.json",
    6: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-6/82c5af999e43d4855f1e1400c2c455c4fb10fe5e/data/processed_data.json",
    12: "https://raw.githubusercontent.com/LIBRA-project/BABY-1L-run-12/31cbfc480c4901c8be3f24f85f8cc24596057c60/data/processed_data.json",
}

RUNS = (1, 2, 3, 4, 6, 12)  # campaign order
TRAIN = (1, 2, 3, 12)  # calibration runs
HELD = (4, 6)  # predicted runs
RESET_BEFORE = {12}  # run 12 starts from a clean system (runs 9-11 are not modelled)

# Sweep gas: (switch day or None, H2 ppm before, H2 ppm after); 0 ppm = pure He.
H2_SCHEDULE = {
    1: (None, 0.0, 0.0),
    2: (None, 0.0, 0.0),
    3: (18.210, 0.0, 1000.0),
    4: (17.369, 1000.0, 35000.0),
    6: (3.224, 35000.0, 0.0),
    12: (None, 36500.0, 36500.0),
}
HAS_SPARGER = {1: False, 2: False, 3: True, 4: True, 6: True, 12: True}
RUN_TEMP_C = {
    1: 627.0,
    2: 627.0,
    3: 634.0,
    4: 634.0,
    6: 634.0,
    12: 685.0,
}  # general.json temperature_salt

# Vessel geometry [m]
CM = 1e-2
R_HEATER = 0.439 * CM  # heater well radius
R_CLLIF = 7.000 * CM  # inner radius of the vessel
R_INCONEL = 7.300 * CM  # outer radius of the vessel
T_BOTTOM = 0.300 * CM  # bottom-plate thickness
T_HEATER_GAP = 0.878 * CM  # melt below the heater tip
T_CLLIF = (6.388 + 0.13022) * CM  # melt depth
T_GAP = 4.605 * CM  # side wall above the melt
T_CAP = 1.422 * CM  # top-cap thickness

_MEMO = {}


def raw(run):
    """processed_data.json of a run: downloaded from RUN_URLS on first use, then read from data/."""
    if run not in _MEMO:
        path = os.path.join(DATA, f"run_{run}.json")
        if not os.path.exists(path):
            import requests

            r = requests.get(RUN_URLS[run], timeout=30)
            r.raise_for_status()
            tmp = f"{path}.tmp.{os.getpid()}"  # several processes may download at once
            with open(tmp, "w") as f:
                json.dump(r.json(), f)
            os.replace(tmp, path)
        with open(path) as f:
            _MEMO[run] = json.load(f)
    return _MEMO[run]


SECONDS_PER = {
    "s": 1.0,
    "second": 1.0,
    "h": 3600.0,
    "hour": 3600.0,
    "d": DAY,
    "day": DAY,
}


def irradiation_segments(run, unit="day"):
    """[(start, stop), ...] of the generator-on windows, in days (unit="day") or seconds (unit="s").
    The LIBRA files give them in seconds from the first irradiation."""
    scale = 1.0 if unit == "s" else 1.0 / DAY
    return [
        (
            i["start_time"]["value"] * SECONDS_PER[i["start_time"]["unit"]] * scale,
            i["stop_time"]["value"] * SECONDS_PER[i["stop_time"]["unit"]] * scale,
        )
        for i in raw(run)["irradiations"]
    ]


def irradiation_seconds(run):
    """Total generator-on time [s]."""
    return sum(b - a for a, b in irradiation_segments(run, unit="s"))


# Runs 4 and 12: the foil analyses are in LIBRA pull requests that are not merged yet.
@functools.cache
def measured_rates():
    """{run: {"rate", "rel_sigma", "source"}}: neutron rates from data/measured_neutron_rates.json."""
    with open(os.path.join(DATA, "measured_neutron_rates.json")) as f:
        return {int(r): m for r, m in json.load(f).items() if not r.startswith("_")}


def neutron_rate(run):
    """Neutron rate [n/s] from the foil measurement: data/measured_neutron_rates.json if the run is
    there, otherwise the run's LIBRA file."""
    if run in measured_rates():
        return measured_rates()[run]["rate"]
    return raw(run)["neutron_rate_used_in_model"]["value"]


def neutron_rate_rel_sigma(run):
    """Relative scatter of the foil measurement behind neutron_rate(run)."""
    if run in measured_rates():
        return measured_rates()[run]["rel_sigma"]
    d = raw(run)["neutron_rate_used_in_model"]
    return d["error"] / d["value"]


def openmc_tbr():
    """OpenMC tritium breeding ratio of runs 1-4 (modelled_TBR, 2.08e-3): the centre of the TBR prior."""
    return raw(1)["modelled_TBR"]["mean"]


def neutrons(run):
    """Neutrons emitted over the run's irradiation."""
    return neutron_rate(run) * irradiation_seconds(run)


def rate_priors():
    """{run: relative sigma} of the log-normal prior on rho_r, centred on 1, for the training runs."""
    return {r: neutron_rate_rel_sigma(r) for r in TRAIN}


def curves(run):
    """Measured cumulative IV and OV release [Bq] at the sampling times [day]."""
    cr = raw(run)["cumulative_tritium_release"]
    return {
        "iv_t": np.asarray(cr["IV"]["sampling_times"]["value"], float),
        "iv": np.asarray(cr["IV"]["total"]["value"], float),
        "ov_t": np.asarray(cr["OV"]["sampling_times"]["value"], float),
        "ov": np.asarray(cr["OV"]["total"]["value"], float),
    }


def t_end(run):
    c = curves(run)
    return float(max(c["iv_t"][-1], c["ov_t"][-1]))


_CYC = None


def cyclotron_windows(run):
    """[(start_day, end_day, logged charge), ...]: beam-on windows of the cyclotron operated next to
    the vessel (data/cyc_windows.json, from the cyclotron log and the run-4 logbook).  The charge is in
    the log's unit (logged "C" = true C / 1440)."""
    global _CYC
    if _CYC is None:
        with open(os.path.join(DATA, "cyc_windows.json")) as f:
            _CYC = {
                int(r): [(float(a), float(b), float(q)) for a, b, q in rows]
                for r, rows in json.load(f).items()
                if not r.startswith("_")
            }
    return _CYC.get(run, [])
