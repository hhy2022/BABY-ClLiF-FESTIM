"""The figures, made from the results of run_all.sh.  Every model curve is FESTIM followed by the
downstream holdup layer.

  fig1_geometry          the vessel, the sweep streams and the two tritium pathways
  fig2_mechanism         the model: breeder side (2-D FESTIM), downstream holdup layer, counting
  fig3_training          the training runs against the measurements
  fig4_prediction        the held-out runs, with the range of the prediction
  fig5_inventory         where the tritium is during the campaign, run by run
  fig6_pathways_run3     the tritium collected in run 3, split by where it comes from
  fig7_parameter_ranges  the profile likelihood of every calibrated parameter

    python figures.py figures
"""

import argparse
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    Polygon,
    Rectangle,
)

import chain
import experimental_data
import festim_output
import likelihood
import parameters

matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument(
    "figures",
    nargs="?",
    default=os.path.join(parameters.ROOT, "figures"),
    help="output folder of run_all.sh (default figures)",
)
FIG = ap.parse_args().figures
W = os.path.join(FIG, "model_output")
BQ = experimental_data.BQ_PER_ATOM

# ------------------------------------------------------------------ shared style
FS_TICK, FS_LABEL, FS_TAG, FS_LEG, FS_SW = 13, 14, 13, 13, 12
plt.rcParams.update({"xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK})
IVC, OVC, IRR, SWC, CYC = "#1B9E77", "#D9601F", "#E8709A", "#8A8A8A", "#7B6FB0"
MK_IV = dict(marker="o", ms=6.5, mfc=IVC, mec=IVC, color=IVC, ls="none")
MK_OV = dict(marker="o", ms=6.5, mfc="white", mec=OVC, mew=1.7, color=OVC, ls="none")
LN_IV, LN_OV = dict(color=IVC, lw=2.2), dict(color=OVC, lw=2.2)
LN_IV_DASH, LN_OV_DASH = dict(color=IVC, lw=1.8, ls=(0, (4, 2.5))), dict(
    color=OVC, lw=1.8, ls=(0, (4, 2.5))
)
GAS_TEXT = {
    1: (None, "He, no sparger"),
    2: (None, "He, no sparger"),
    3: ("He $\\rightarrow$ 0.1% H$_2$", None),
    4: ("0.1% $\\rightarrow$ 3.5% H$_2$", None),
    6: ("3.5% H$_2$ $\\rightarrow$ He", None),
    12: (None, "3.65% H$_2$, 685 °C"),
}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {parameters.rel(os.path.join(FIG, name))}.png/.pdf")


def beam_blocks(run, gap_min=10.0):
    """Cyclotron beam windows, those closer than gap_min minutes merged (for drawing)."""
    out = []
    for s, e, _q in sorted(experimental_data.cyclotron_windows(run)):
        if out and s - out[-1][1] <= gap_min / 1440.0:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, max(e, s + 1 / 1440)) for s, e in out]


def decorate(ax, run, top):
    """Irradiation bands, gas switch, gas label, measured points, axes."""
    c = experimental_data.curves(run)
    for s, e in experimental_data.irradiation_segments(run):
        ax.axvspan(s, e, color=IRR, alpha=0.28, lw=0, zorder=0)
    k = c["iv_t"] > 0
    ax.plot(c["iv_t"][k], c["iv"][k], zorder=4, **MK_IV)
    k = c["ov_t"] > 0
    ax.plot(c["ov_t"][k], c["ov"][k], zorder=4, **MK_OV)
    ax.set_ylabel("cumulative recovery [Bq]", fontsize=FS_LABEL)
    ax.set_xlim(0, 1.03 * experimental_data.t_end(run))
    ax.set_ylim(0, 1.12 * max(top, c["iv"].max(), c["ov"].max()))
    sw_label, cond = GAS_TEXT[run]
    sw = experimental_data.H2_SCHEDULE[run][0]
    if sw is not None:
        xmax = ax.get_xlim()[1]
        right = sw < 0.6 * xmax
        ax.axvline(sw, color=SWC, ls=(0, (5, 4)), lw=1.4, zorder=1)
        ax.text(
            sw + (0.012 if right else -0.012) * xmax,
            0.965,
            sw_label,
            transform=ax.get_xaxis_transform(),
            fontsize=FS_SW,
            color=SWC,
            ha="left" if right else "right",
            va="top",
        )
    if cond:
        ax.text(
            0.97,
            0.50,
            cond,
            transform=ax.transAxes,
            fontsize=FS_SW,
            color=SWC,
            ha="right",
            va="center",
        )
    ax.grid(alpha=0.18)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def legend(fig, extra, ncol=4):
    h = [
        Line2D([], [], **MK_IV),
        Line2D([], [], **LN_IV),
        Line2D([], [], **MK_OV),
        Line2D([], [], **LN_OV),
    ]
    lab = ["IV measured", "IV model", "OV measured", "OV model"]
    fig.legend(
        h + [x[0] for x in extra],
        lab + [x[1] for x in extra],
        fontsize=FS_LEG,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=2.0,
        handlelength=2.2,
        frameon=True,
        fancybox=False,
        edgecolor="#BDBDBD",
        framealpha=1.0,
    )


# ------------------------------------------------------------------ model output
FINAL = None  # the final parameter set
P = None  # its FESTIM chain with the holdup layer
NO_CYC = {}  # the same model with the run's own cyclotron source switched off
BAND = []  # FESTIM at the prediction-profile extremes


def load_model_output():
    """Read the FESTIM chains written by run_all.sh (figures 3-7; figures 1 and 2 are drawings)."""
    global FINAL, P
    FINAL = parameters.read_params(os.path.join(W, "festim", "params_refit.txt"))
    P = chain.forward_festim(FINAL, os.path.join(W, "festim"))
    for r in (3, 4):
        d = os.path.join(W, f"festim_no_cyclotron_run{r}")
        if os.path.exists(os.path.join(d, f"run_{r}", "run_complete")):
            NO_CYC[r] = chain.forward_festim(FINAL, d, (r,))[r]
    BAND[:] = [P]
    for d in sorted(glob.glob(os.path.join(W, "prediction_ranges", "festim", "*"))):
        if os.path.exists(os.path.join(d, "run_12", "run_complete")):
            th = parameters.read_params(
                os.path.join(
                    W, "prediction_ranges", "params", os.path.basename(d) + ".txt"
                )
            )
            BAND.append(chain.forward_festim(th, d))


# ================================================================== Fig. 1: geometry
def fig1_geometry():
    """The vessel to scale (cm, baby_mesh.py), the sweep streams and the two tritium pathways."""
    C_SALT, C_INCONEL, C_GAS, C_OVGAS, C_OVEDGE = (
        "#F2B24A",
        "#7FA6CE",
        "#E9F2F7",
        "#E7F0E7",
        "#9DBE9D",
    )
    C_HEATER, C_HEATER_E, C_FLOW, EDGE = "#DCDCDC", "#CBCBCB", "#6E6E6E", "#2B2B2B"
    cm = experimental_data.CM  # drawn in cm
    rh = experimental_data.R_HEATER / cm
    rc = experimental_data.R_CLLIF / cm
    ri = experimental_data.R_INCONEL / cm
    zsb = experimental_data.T_BOTTOM / cm  # bottom plate
    zhb = zsb + experimental_data.T_HEATER_GAP / cm  # heater tip
    zst = zsb + experimental_data.T_CLLIF / cm  # free surface
    zgt = zst + experimental_data.T_GAP / cm  # top of the headspace
    zct = zgt + experimental_data.T_CAP / cm  # top of the cap
    OVX, OVB, OVT = 9.2, -2.0, 13.8
    fig, ax = plt.subplots(figsize=(10.6, 10.9))

    def arrow(x0, z0, x1, z1, color, lw=2.2, ms=16, z=8):
        ax.add_patch(
            FancyArrowPatch(
                (x0, z0),
                (x1, z1),
                arrowstyle="-|>",
                mutation_scale=ms,
                color=color,
                lw=lw,
                zorder=z,
                shrinkA=0,
                shrinkB=0,
            )
        )

    # outer vessel, also swept
    ax.add_patch(
        Rectangle(
            (-OVX, OVB),
            2 * OVX,
            OVT - OVB,
            facecolor=C_OVGAS,
            edgecolor=C_OVEDGE,
            lw=1.1,
            ls=(0, (6, 3)),
            zorder=0,
        )
    )
    ax.add_patch(Rectangle((-rc, zst), 2 * rc, zgt - zst, facecolor=C_GAS, zorder=1))
    ax.add_patch(Rectangle((-rc, zsb), 2 * rc, zst - zsb, facecolor=C_SALT, zorder=2))
    metal = [
        (-ri, 0.0),
        (-ri, zct),
        (-rh, zct),
        (-rh, zgt),
        (-rc, zgt),
        (-rc, zsb),
        (rc, zsb),
        (rc, zgt),
        (rh, zgt),
        (rh, zct),
        (ri, zct),
        (ri, 0.0),
    ]
    ax.add_patch(
        Polygon(
            metal,
            closed=True,
            joinstyle="round",
            facecolor=C_INCONEL,
            edgecolor=EDGE,
            lw=1.1,
            zorder=3,
        )
    )
    ax.plot([-rc, rc], [zst, zst], color="#9C6B12", lw=2.0, zorder=4)
    ax.add_patch(
        Rectangle(
            (-rh, zhb),
            2 * rh,
            zct + 0.7 - zhb,
            facecolor=C_HEATER,
            edgecolor=C_HEATER_E,
            lw=0.9,
            zorder=5,
        )
    )

    BOX_Z, BOX_H = 13.95, 1.15
    sweep = dict(color=C_FLOW, lw=2.6, ms=20, z=6)
    arrow(-5.6, BOX_Z - 0.03, -5.6, 11.15, **sweep)  # sweep in -> IV headspace
    arrow(3.1, 11.15, 3.1, BOX_Z - 0.03, **sweep)  # IV headspace -> IV collection
    arrow(-8.2, BOX_Z - 0.03, -8.2, 12.15, **sweep)  # sweep in -> OV enclosure
    arrow(7.7, 12.15, 7.7, BOX_Z - 0.03, **sweep)  # OV enclosure -> OV collection
    # salt -> headspace
    arrow(-2.6, zst + 0.05, -2.6, zst + 2.15, IVC, lw=5.0, ms=30, z=6)
    arrow(rc - 0.05, 8.6, 5.0, 8.6, IVC, lw=5.0, ms=30, z=6)  # inner wall -> headspace
    # wall -> OV enclosure
    arrow(rc - 0.6, 3.6, ri + 1.15, 3.6, OVC, lw=5.0, ms=30, z=6)

    ax.text(
        -3.6,
        3.4,
        "ClLiF salt",
        fontsize=27,
        fontweight="bold",
        ha="center",
        va="center",
        color="#6B4A08",
        zorder=7,
    )
    ax.text(
        -7.70,
        7.6,
        "Inconel-625",
        fontsize=17,
        rotation=90,
        ha="center",
        va="center",
        color="#2F5076",
        zorder=7,
    )
    ax.text(
        0.0,
        8.0,
        "heater",
        fontsize=14,
        rotation=90,
        ha="center",
        va="center",
        color="#AFAFAF",
        zorder=7,
    )
    ax.text(
        3.6,
        10.0,
        "IV headspace",
        fontsize=19,
        ha="center",
        va="center",
        color="#4A6B7C",
        zorder=7,
    )
    ax.text(
        -4.9,
        7.95,
        "free-surface /\nsparging release",
        fontsize=16.5,
        ha="center",
        va="center",
        color=IVC,
        zorder=7,
    )
    ax.text(
        4.75,
        7.55,
        "wall release to IV",
        fontsize=16.5,
        ha="center",
        va="center",
        color=IVC,
        zorder=7,
    )
    ax.text(
        6.6,
        2.45,
        "wall release to OV",
        fontsize=16.5,
        ha="right",
        va="center",
        color="#8A3A0E",
        zorder=7,
    )
    ax.text(
        -8.85,
        -1.05,
        "OV enclosure",
        fontsize=19,
        ha="left",
        va="center",
        color="#9C4A14",
        zorder=7,
    )

    def collection_box(cx, color, text, w=3.9):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, BOX_Z),
                w,
                BOX_H,
                boxstyle="round,pad=0.02,rounding_size=0.14",
                facecolor="white",
                edgecolor=color,
                lw=1.9,
                zorder=8,
            )
        )
        ax.text(
            cx,
            BOX_Z + BOX_H / 2,
            text,
            color=color,
            fontsize=16.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=9,
        )

    collection_box(-6.9, C_FLOW, "He / H₂ sweep")
    collection_box(3.1, IVC, "IV collection")
    collection_box(7.7, OVC, "OV collection")
    for x in (-5.4, -1.8, 1.8, 5.4):
        arrow(x, -3.30, x, OVB - 0.12, C_FLOW, lw=2.6, ms=20, z=6)
    ax.text(
        0.0,
        -3.85,
        "neutrons",
        fontsize=20,
        ha="center",
        va="center",
        color=C_FLOW,
        zorder=7,
    )
    ax.set_xlim(-10.6, 10.6)
    ax.set_ylim(-4.5, 15.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig1_geometry")


# ================================================================== Fig. 2: mechanism
def fig2_mechanism():
    """The model as a diagram.  The processes are numbered 1-8 on the figure; the key is in the caption:
    1 source in the melt, 2 sparging, 3 free-surface release, 4 melt|Inconel interface, 5 release from
    the outer wall, 6 and 7 IV and OV line holdup, 8 the state carried to the next run.
    """
    C_SALT, C_INCONEL, C_GAS = "#F2B24A", "#7FA6CE", "#E9F2F7"
    C_OVGAS, C_OVEDGE, C_HEATER, C_HEATER_E = "#E7F0E7", "#9DBE9D", "#DCDCDC", "#CBCBCB"
    C_IFACE, C_ZERO, C_MEM, C_TXT, C_MUTED = (
        "#B22222",
        "#8C8C8C",
        "#6B4FA0",
        "#222222",
        "#6A6A6A",
    )
    FS_HDR, FS_REG, FS_LAB, FS_NUM = 12.5, 11.0, 10.0, 9.0
    DASH, DASH_MEM = (0, (4, 2)), (0, (2.5, 1.5))
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    def arrow(p0, p1, color, lw=1.8, ls="-", ms=11, z=6):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=ms,
                color=color,
                lw=lw,
                linestyle=ls,
                shrinkA=0,
                shrinkB=0,
                zorder=z,
            )
        )

    def seg(pts, color, lw=1.8, ls="-", z=6, cap="butt"):
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=color,
            lw=lw,
            ls=ls,
            zorder=z,
            solid_capstyle=cap,
            solid_joinstyle="round",
        )

    def num(x, y, n, color):
        ax.add_patch(
            Circle((x, y), 0.31, facecolor=color, edgecolor="white", lw=0.9, zorder=12)
        )
        ax.text(
            x,
            y - 0.01,
            str(n),
            ha="center",
            va="center",
            fontsize=FS_NUM,
            fontweight="bold",
            color="white",
            zorder=13,
        )

    def rbox(x0, y0, w, h, text, ec, fc="white", fs=FS_LAB, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=fc,
                edgecolor=ec,
                lw=1.3,
                zorder=5,
            )
        )
        ax.text(
            x0 + w / 2,
            y0 + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color=ec,
            fontweight="bold" if bold else "normal",
            zorder=6,
            linespacing=1.2,
        )

    # breeder side (cross-section, wall and cap thickness exaggerated)
    EX0, EX1, EY0, EY1 = 0.3, 7.9, 3.3, 10.5  # OV enclosure
    VX0, VX1, VY0, VY1 = 1.4, 6.8, 4.0, 9.7  # vessel outer
    TW, TC = 0.24, 0.62
    IX0, IX1 = VX0 + TW, VX1 - TW
    YM0, YFS, YCAP = VY0 + TW, 6.75, VY1 - TC  # melt bottom, free surface, cap inner
    HX0, HX1 = 3.95, 4.25  # heater well
    ax.add_patch(
        Rectangle(
            (EX0, EY0),
            EX1 - EX0,
            EY1 - EY0,
            facecolor=C_OVGAS,
            edgecolor=C_OVEDGE,
            lw=1.0,
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle(
            (VX0, VY0),
            VX1 - VX0,
            VY1 - VY0,
            facecolor=C_INCONEL,
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (IX0, YFS),
            IX1 - IX0,
            YCAP - YFS,
            facecolor=C_GAS,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (IX0, YM0),
            IX1 - IX0,
            YFS - YM0,
            facecolor=C_SALT,
            edgecolor="none",
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (HX0, YM0 + 0.55),
            HX1 - HX0,
            VY1 + 0.35 - (YM0 + 0.55),
            facecolor=C_HEATER,
            edgecolor=C_HEATER_E,
            lw=0.6,
            zorder=5,
        )
    )
    # boundaries, coloured by where tritium crossing them goes
    seg([(IX0, YFS), (HX0, YFS)], IVC, lw=2.0, z=4, cap="round")  # free surface -> IV
    seg([(HX1, YFS), (IX1, YFS)], IVC, lw=2.0, z=4, cap="round")
    # inner wall above the melt: closed
    seg([(IX0, YFS), (IX0, YCAP), (HX0, YCAP)], C_ZERO, lw=1.6, z=4, cap="round")
    seg([(HX1, YCAP), (IX1, YCAP), (IX1, YFS)], C_ZERO, lw=1.6, z=4, cap="round")
    # melt | Inconel
    seg(
        [(IX0, YFS), (IX0, YM0), (IX1, YM0), (IX1, YFS)],
        C_IFACE,
        lw=1.5,
        z=4,
        cap="round",
    )
    # outer surface -> OV
    ax.add_patch(
        Rectangle(
            (VX0, VY0),
            VX1 - VX0,
            VY1 - VY0,
            facecolor="none",
            edgecolor=OVC,
            lw=1.8,
            joinstyle="miter",
            zorder=4,
        )
    )
    ax.text(
        2.75,
        5.0,
        "ClLiF salt",
        fontsize=FS_REG,
        fontweight="bold",
        color="#6B4A08",
        ha="center",
        va="center",
        zorder=7,
    )
    ax.text(
        2.78,
        8.65,
        "headspace",
        fontsize=FS_LAB,
        color="#7A8F9C",
        ha="center",
        va="center",
        zorder=7,
    )
    ax.text(
        2.7,
        VY1 - TC / 2,
        "Inconel 625",
        fontsize=FS_LAB,
        fontweight="bold",
        color="#2F5076",
        ha="center",
        va="center",
        zorder=7,
    )
    ax.text(
        EX0 + 0.15,
        EY0 + 0.25,
        "OV enclosure",
        fontsize=FS_LAB,
        color="#9C4A14",
        ha="left",
        va="center",
        zorder=7,
    )

    XOUT, XSP = 5.0, 6.05
    XN, XP0, XPW, XB0, XBW = 9.55, 10.0, 2.55, 13.4, 2.6
    Y_IV, Y_OV = 11.2, 5.35
    num(2.2, 5.85, 1, "#6B4A08")  # 1 source in the melt
    # 2 sparging, booked straight into the IV stream
    arrow((XSP, 6.1), (XSP, Y_IV), IVC, lw=1.6, ls=DASH)
    num(XSP, 6.1, 2, IVC)
    # 3 free surface -> IV line
    seg([(XOUT, YFS), (XOUT, Y_IV), (XN, Y_IV)], IVC, lw=2.2)
    num(XOUT, YFS, 3, IVC)
    X6 = IX1 - 1.3  # 4 melt | Inconel interface
    arrow((X6, Y_OV), (IX1, Y_OV), C_IFACE, lw=1.8)
    num(X6, Y_OV, 4, C_IFACE)
    seg([(VX1, Y_OV), (XN, Y_OV)], OVC, lw=2.2)  # 5 wall and outer surface -> OV line
    num(VX1 + 0.55, Y_OV, 5, OVC)

    # downstream: each line with its holdup pool (6, 7), then the bubblers
    YP_IV, YP_OV, H_P = 9.25, 3.4, 1.0
    XR, EDGE = XP0 + XPW - 0.45, 0.04
    for y, yp, n, color, name, fc in (
        (Y_IV, YP_IV, 6, IVC, "IV", "#E4F3EE"),
        (Y_OV, YP_OV, 7, OVC, "OV", "#FBEBE1"),
    ):
        arrow((XN, y), (XB0, y), color, lw=2.2)
        seg([(XN, y), (XN, yp + H_P / 2)], color, lw=1.6)
        arrow((XN, yp + H_P / 2), (XP0 - EDGE, yp + H_P / 2), color, lw=1.6, ms=9)
        num(XN, 0.5 * (y + yp + H_P / 2), n, color)
        rbox(XP0, yp, XPW, H_P, f"{name}-line holdup", color)
        arrow(
            (XR, yp + (H_P + EDGE if y > yp else -EDGE)),
            (XR, y),
            color,
            lw=1.6,
            ms=9,
            z=4,
        )
        rbox(
            XB0,
            y - 0.6,
            XBW,
            1.2,
            f"{name} bubbler",
            color,
            fc=fc,
            fs=FS_REG,
            bold=True,
        )

    # 8 the state carried to the next run (in time, not in space)
    XMEM, YMEM, XV, X_END = 12.85, 2.55, 0.5 * (EX0 + EX1), 13.75
    seg([(XV, EY0), (XV, YMEM), (X_END - 0.3, YMEM)], C_MEM, lw=1.3, ls=DASH_MEM, z=4)
    seg(
        [(XP0 + XPW + EDGE, YP_IV + H_P / 2), (XMEM, YP_IV + H_P / 2), (XMEM, YMEM)],
        C_MEM,
        lw=1.3,
        ls=DASH_MEM,
        z=4,
    )
    seg(
        [(XP0 + XPW + EDGE, YP_OV + H_P / 2), (XMEM, YP_OV + H_P / 2)],
        C_MEM,
        lw=1.3,
        ls=DASH_MEM,
        z=4,
    )
    arrow((X_END - 0.3, YMEM), (X_END, YMEM), C_MEM, lw=1.3, ms=9)
    ax.text(
        X_END + 0.1,
        YMEM,
        "next run",
        fontsize=FS_LAB,
        color=C_MEM,
        ha="left",
        va="center",
        zorder=7,
    )
    num(8.2, YMEM, 8, C_MEM)

    for x, t, s in (
        (4.1, "breeder side", "2-D FESTIM"),
        (10.85, "downstream\ncollection system", "0-D holdup"),
        (14.7, "counting", ""),
    ):
        ax.text(
            x,
            12.55,
            t,
            fontsize=FS_HDR,
            fontweight="bold",
            color=C_TXT,
            ha="center",
            va="bottom",
            linespacing=1.1,
        )
        if s:
            ax.text(
                x,
                12.25,
                s,
                fontsize=FS_LAB,
                style="italic",
                color=C_MUTED,
                ha="center",
                va="center",
            )
    for xs in (8.55, 13.25):
        ax.plot([xs, xs], [2.9, 13.4], color="#D5D5D5", lw=0.8, zorder=0)
    ax.set_xlim(0.0, 16.1)
    ax.set_ylim(2.1, 13.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    save(fig, "fig2_mechanism")


# ================================================================== Figs. 3 and 4: training and prediction
def run_panel(spec, tag, run, band):
    inner = spec.subgridspec(2, 1, height_ratios=[6.0, 0.8], hspace=0.06)
    ax = plt.gcf().add_subplot(inner[0])
    blocks = beam_blocks(run)
    strip = plt.gcf().add_subplot(inner[1], sharex=ax) if blocks else None
    t, iv, ov = P[run]
    top = iv.max()
    if band and len(BAND) > 1:
        for i, col in ((1, IVC), (2, OVC)):
            Y = np.array([np.interp(t, Q[run][0], Q[run][i]) for Q in BAND])
            ax.fill_between(
                t, Y.min(0), Y.max(0), color=col, alpha=0.16, lw=0, zorder=1
            )
        top = max(top, max(Q[run][1].max() for Q in BAND))
    if run in NO_CYC:
        ax.plot(NO_CYC[run][0], NO_CYC[run][1], zorder=2, **LN_IV_DASH)
        ax.plot(NO_CYC[run][0], NO_CYC[run][2], zorder=2, **LN_OV_DASH)
    ax.plot(t, iv, zorder=3, **LN_IV)
    ax.plot(t, ov, zorder=3, **LN_OV)
    decorate(ax, run, top)
    ax.set_title(tag, fontsize=FS_TAG, fontweight="bold")
    if strip is None:
        ax.set_xlabel("time [day]", fontsize=FS_LABEL)
    else:
        ax.tick_params(labelbottom=False)
        for s, e in blocks:
            strip.axvspan(s, e, facecolor=CYC, edgecolor=CYC, alpha=0.75, lw=1.0)
        strip.set_xlabel("time [day]", fontsize=FS_LABEL)
        strip.set_ylim(0, 1)
        strip.set_yticks([])
        strip.set_ylabel(
            "cyclotron\nbeam-on",
            fontsize=FS_LABEL - 2,
            color=CYC,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )
        for sp in ("top", "right", "left"):
            strip.spines[sp].set_visible(False)


def fig3_training():
    fig = plt.figure(figsize=(10.6, 8.9))
    g = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)
    for i, (tag, r) in enumerate(
        (("(a) Run 1", 1), ("(b) Run 2", 2), ("(c) Run 3", 3), ("(d) Run 12", 12))
    ):
        run_panel(g[i // 2, i % 2], tag, r, band=False)
    legend(
        fig,
        (
            [
                (Line2D([], [], **LN_IV_DASH), "IV, no cyclotron"),
                (Line2D([], [], **LN_OV_DASH), "OV, no cyclotron"),
            ]
            if 3 in NO_CYC
            else []
        ),
    )
    fig.subplots_adjust(top=0.875, bottom=0.07, left=0.10, right=0.985)
    save(fig, "fig3_training")


def fig4_prediction():
    fig = plt.figure(figsize=(12.0, 5.6))
    g = fig.add_gridspec(1, 2, wspace=0.25, width_ratios=[1.5, 1.0])
    run_panel(g[0], "(a) Run 4", 4, band=True)
    run_panel(g[1], "(b) Run 6", 6, band=True)
    extra = [
        (Patch(color=IVC, alpha=0.16), "IV admissible range"),
        (Patch(color=OVC, alpha=0.16), "OV admissible range"),
    ]
    if 4 in NO_CYC:
        extra += [
            (Line2D([], [], **LN_IV_DASH), "IV, no run-4 cyclotron"),
            (Line2D([], [], **LN_OV_DASH), "OV, no run-4 cyclotron"),
        ]
    legend(fig, extra)
    fig.subplots_adjust(top=0.78, bottom=0.10, left=0.08, right=0.985)
    save(fig, "fig4_prediction")


# ================================================================== Fig. 5: inventory
def pool_chain(stop_before=None):
    """Holdup layer along the FESTIM chain: per run (t, pool record); the state at the start of
    `stop_before` if given."""
    out, state = {}, None
    for r in experimental_data.RUNS:
        if state is None or r in experimental_data.RESET_BEFORE:
            state = dict(N_iv=0.0, N_ov=0.0)
        if r == stop_before:
            return out, state
        rec = []
        res = festim_output.streams(os.path.join(W, "festim", f"run_{r}"))
        chain.apply_layer(r, res, FINAL, state, record=rec)
        out[r] = (res["t"], np.array(rec))
    return out, state


def fig5_inventory():
    """One row per quantity; each run on its own time axis (days since its first irradiation), runs side by side."""
    pools, _ = pool_chain()
    data = {}
    for r in experimental_data.RUNS:
        d = os.path.join(W, "festim", f"run_{r}")
        salt = np.loadtxt(
            os.path.join(d, "inventory_salt.csv"), delimiter=",", skiprows=1
        )
        wall = np.loadtxt(
            os.path.join(d, "inventory_wall.csv"), delimiter=",", skiprows=1
        )
        t, rec = pools[r]
        data[r] = dict(
            T=float(t[-1]),
            salt=(salt[:, 0] / experimental_data.DAY, salt[:, 1] * BQ),
            wall=(wall[:, 0] / experimental_data.DAY, wall[:, 1] * BQ),
            iv=(rec[:, 0], rec[:, 1] * BQ),
            ov=(rec[:, 0], rec[:, 2] * BQ),
        )
    C_MELT, C_WALL = "#eda100", "#2a78d6"
    INK, INK2, GRID, SURF, C_FAINT = (
        "#0b0b0b",
        "#52514e",
        "#e4e3df",
        "#fcfcfb",
        "#a8a7a3",
    )
    GAS_FILL = {0.0: None, 1000.0: "#f4f3f0", 35000.0: "#ebeae5", 36500.0: "#ebeae5"}
    GAS_NAME = {
        0.0: "He",
        1000.0: "0.1 % H$_2$",
        35000.0: "3.5 % H$_2$",
        36500.0: "3.65 % H$_2$",
    }
    XTICKS = {
        1: [0, 10, 20, 30],
        2: [0, 20, 40, 60],
        3: [0, 10, 20, 30, 40],
        4: [0, 10, 20],
        6: [0, 1, 2, 3],
        12: [0, 2, 4, 6],
    }
    GAP, GAP12 = 0.06, 0.22  # run r occupies [X0[r], X0[r] + 1]
    X0, x = {}, 0.0
    for r in experimental_data.RUNS:
        if r in experimental_data.RESET_BEFORE:
            x += GAP12 - GAP
        X0[r] = x
        x += 1.0 + GAP
    XMAX = x - GAP

    def xm(r, t):
        return X0[r] + np.asarray(t, float) / data[r]["T"]

    def gas_segments(r):
        sw, a, b = experimental_data.H2_SCHEDULE[r]
        T = data[r]["T"]
        return [(0.0, T, a)] if sw is None else [(0.0, sw, a), (sw, T, b)]

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    ):
        fig = plt.figure(figsize=(11.0, 6.6), facecolor=SURF)
        gs = fig.add_gridspec(
            4,
            1,
            height_ratios=[1.15, 0.10, 0.10, 1.0],
            hspace=0.10,
            left=0.075,
            right=0.995,
            top=0.885,
            bottom=0.085,
        )
        top = fig.add_subplot(gs[0])
        strip = fig.add_subplot(gs[1], sharex=top)
        bot = fig.add_subplot(gs[3], sharex=top)
        top.tick_params(axis="x", labelbottom=False)
        strip.tick_params(axis="x", labelbottom=False)
        for ax in (top, bot):
            ax.set_facecolor(SURF)
            ax.grid(axis="y", color=GRID, lw=0.6, zorder=1)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        strip.set_facecolor(SURF)
        ticks, tlabels = [], []
        for r in experimental_data.RUNS:
            T = data[r]["T"]
            for ax in (top, bot):
                for a, b, ppm in gas_segments(r):
                    if GAS_FILL[ppm]:
                        ax.axvspan(
                            xm(r, a), xm(r, b), color=GAS_FILL[ppm], lw=0, zorder=0
                        )
            top.plot(
                xm(r, data[r]["salt"][0]),
                data[r]["salt"][1],
                color=C_MELT,
                lw=1.7,
                zorder=4,
            )
            top.plot(
                xm(r, data[r]["wall"][0]),
                data[r]["wall"][1],
                color=C_WALL,
                lw=1.7,
                zorder=4,
            )
            bot.plot(
                xm(r, data[r]["iv"][0]), data[r]["iv"][1], color=IVC, lw=1.7, zorder=4
            )
            bot.plot(
                xm(r, data[r]["ov"][0]), data[r]["ov"][1], color=OVC, lw=1.7, zorder=4
            )
            ticks += [xm(r, d) for d in XTICKS[r]]
            tlabels += [str(d) for d in XTICKS[r]]
            # source windows, widened to >= 1.5 % of the run's width so an hour-long window shows
            for y, wins, color in (
                (1.0, experimental_data.irradiation_segments(r), IRR),
                (0.0, beam_blocks(r), CYC),
            ):
                for a, b in wins:
                    c, w = xm(r, 0.5 * (a + b)), max((b - a) / T, 0.015)
                    strip.add_patch(
                        plt.Rectangle(
                            (c - 0.5 * w, y - 0.3),
                            w,
                            0.6,
                            facecolor=color,
                            edgecolor="none",
                            zorder=4,
                        )
                    )
            top.text(
                X0[r] + 0.5,
                1.10,
                f"Run {r}",
                transform=top.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color=INK,
            )
            sw, g_a, g_b = experimental_data.H2_SCHEDULE[r]
            gas = (
                GAS_NAME[g_a]
                if sw is None
                else f"{GAS_NAME[g_a]} $\\rightarrow$ {GAS_NAME[g_b]}"
            )
            top.text(
                X0[r] + 0.5,
                1.025,
                gas,
                transform=top.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK2,
            )
        bot.set_xticks(ticks)
        bot.set_xticklabels(tlabels)
        bot.set_xlim(-0.01, XMAX + 0.01)
        top.set_ylim(
            0,
            1.08
            * max(
                max(data[r][k][1].max() for r in experimental_data.RUNS)
                for k in ("salt", "wall")
            ),
        )
        bot.set_ylim(
            0,
            1.15
            * max(
                max(data[r][k][1].max() for r in experimental_data.RUNS)
                for k in ("iv", "ov")
            ),
        )
        for r0, r1 in zip(experimental_data.RUNS[:-1], experimental_data.RUNS[1:]):
            xs = 0.5 * (X0[r0] + 1.0 + X0[r1])
            reset = r1 in experimental_data.RESET_BEFORE
            for ax in (top, strip, bot):
                ax.axvline(
                    xs,
                    color=INK2 if reset else C_FAINT,
                    lw=1.4 if reset else 0.8,
                    ls=(0, (3, 2.5)) if reset else (0, (4, 3)),
                    zorder=3,
                )
            if reset:
                top.text(
                    xs + 0.03,
                    1.10,
                    "state\nreset",
                    transform=top.get_xaxis_transform(),
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    color=INK2,
                    linespacing=1.0,
                )
            else:  # inventories carried to the next run
                top.text(
                    xs,
                    1.115,
                    "→",
                    transform=top.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=C_FAINT,
                )
        strip.set_ylim(-0.6, 1.6)
        strip.set_yticks([1.0, 0.0])
        strip.set_yticklabels(["irradiation", "cyclotron"], fontsize=6.5, color=INK2)
        strip.tick_params(axis="both", length=0)
        for y in (1.0, 0.0):
            strip.axhline(y, color=GRID, lw=0.6, zorder=1)
        for sp in strip.spines.values():
            sp.set_visible(False)
        fig.text(
            0.013,
            0.5 * (top.get_position().y1 + bot.get_position().y0),
            "Tritium inventory (Bq)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            color=INK,
        )
        pt = top.get_position()
        fig.text(
            0.5 * (pt.x0 + pt.x1),
            pt.y1 + 0.19 * (pt.y1 - pt.y0) + 0.012,
            "(a) Breeder-side inventory",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
        )
        fig.text(
            0.5 * (pt.x0 + pt.x1),
            0.5 * (strip.get_position().y0 + bot.get_position().y1),
            "(b) Downstream holdup",
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK,
        )
        fig.text(
            0.535,
            0.02,
            "days since the first irradiation of each run",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
        top.legend(
            handles=[
                Line2D([], [], color=C_MELT, lw=1.7, label="ClLiF salt"),
                Line2D([], [], color=C_WALL, lw=1.7, label="Inconel 625"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.012, 0.98),
            frameon=False,
            fontsize=7.5,
            handlelength=1.6,
        )
        bot.legend(
            handles=[
                Line2D([], [], color=IVC, lw=1.7, label="IV holdup"),
                Line2D([], [], color=OVC, lw=1.7, label="OV holdup"),
            ],
            loc="upper left",
            bbox_to_anchor=(0.012, 0.98),
            frameon=False,
            fontsize=7.5,
            handlelength=1.6,
        )
        save(fig, "fig5_inventory")
    for r in experimental_data.RUNS:
        D = data[r]
        print(
            f"   run {r:>2}: end melt {D['salt'][1][-1]:6.2f}  wall {D['wall'][1][-1]:6.2f}  "
            f"IV pool {D['iv'][1][-1]:5.2f}  OV pool {D['ov'][1][-1]:5.2f} Bq"
        )


# ================================================================== Fig. 6: run 3 by origin
def fig6_pathways(run=3):
    """Cumulative recovery of one run, stacked by where it comes from.

    IV: sparging and free surface collected directly ((1 - alpha_IV) of their release), IV-line holdup
    deposited in this run, and IV-line holdup left from the previous run.  OV: holdup left from the
    previous run, tritium that was in the wall before the run, and tritium bred in the run.

    The holdup layer is linear in its pools and inputs, so the part left from the previous run is the
    run with minus the run without those pools, and each stream can be passed through the layer on its
    own.  The wall release is split by tagging what the wall holds at the start of the run; every release
    takes old and new tritium in proportion to what the wall holds.  Diffusion takes about 1.5 h across
    the side wall and 1.5 days across the cap, so the wall is well mixed on the scale of a run.
    """
    prev = experimental_data.RUNS[experimental_data.RUNS.index(run) - 1]
    _, inherited = pool_chain(stop_before=run)
    d = os.path.join(W, "festim", f"run_{run}")
    res = festim_output.streams(d)
    t = res["t"]
    dts = np.diff(t) * experimental_data.DAY
    rec = []
    civ, cov = chain.apply_layer(run, res, FINAL, dict(inherited), record=rec)
    civ0, cov0 = chain.apply_layer(run, res, FINAL, dict(N_iv=0.0, N_ov=0.0))
    rec = np.array(rec)

    def column(name):  # per step, aligned with res (0 for the initial point)
        a = np.loadtxt(os.path.join(d, name), delimiter=",", skiprows=1, ndmin=2)
        return np.concatenate([[0.0], a[:, 1]])

    f_sp = (
        column("flux_sparge.csv")
        if os.path.exists(os.path.join(d, "flux_sparge.csv"))
        else np.zeros_like(t)
    )
    f_wall = sum(column(f"flux_outer_{k}.csv") for k in ("bottom", "side", "top"))
    n_wall = column("inventory_wall.csv")
    n_wall[0] = np.loadtxt(
        os.path.join(W, "festim", f"run_{prev}", "inventory_wall.csv"),
        delimiter=",",
        skiprows=1,
    )[-1, 1]
    f_inh, n_inh = np.zeros_like(f_wall), n_wall[0]
    for i in range(1, len(t)):
        rel = f_wall[i] * dts[i - 1]
        held = n_wall[i] + rel  # what the wall held before this step's release
        part = rel * min(n_inh / held, 1.0) if held > 0 else 0.0
        f_inh[i] = part / dts[i - 1]
        n_inh -= part
    _, cov_wall_inh = chain.apply_layer(
        run,
        dict(res, rate_iv=np.zeros_like(t), rate_ov=f_inh),
        FINAL,
        dict(N_iv=0.0, N_ov=0.0),
    )

    def cumulative(r):
        return np.concatenate([[0.0], np.cumsum(r[1:] * dts)]) * BQ

    iv_model, ov_model = civ * BQ, cov * BQ
    iv_sp = (1.0 - FINAL["alpha_iv"]) * cumulative(f_sp)
    iv_pool_inh = (civ - civ0) * BQ
    iv_pool_own = np.concatenate([[0.0], np.cumsum(rec[:, 3])]) * BQ - iv_pool_inh
    # closes the stack on the layer's own total
    iv_fs = iv_model - iv_sp - iv_pool_inh - iv_pool_own
    ov_pool_inh = (cov - cov0) * BQ
    ov_wall_inh = cov_wall_inh * BQ
    ov_bred = cov0 * BQ - ov_wall_inh

    c = experimental_data.curves(run)
    sw, ppm_a, ppm_b = experimental_data.H2_SCHEDULE[run]
    T_END = float(t[-1])
    C_SP, C_FS, C_POOL_INH, C_POOL = "#2F5D8C", "#A9C8E6", "#6B5F82", "#C4BCD3"
    C_WALL_INH, C_BRED = "#B9695B", "#9CB8A2"
    INK, INK2, GRID, SHADE_H2 = "#000000", "#555555", "#E3E3E3", "#F5F5F5"
    GAS = {
        0.0: "He",
        1000.0: "0.1 % H$_2$",
        35000.0: "3.5 % H$_2$",
        36500.0: "3.65 % H$_2$",
    }
    FS, FS_S = 8.5, 7.5
    style = {
        "font.size": FS,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "mathtext.default": "regular",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(6.7, 6.0), facecolor="white")
        gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[1.0, 0.115, 1.0],
            hspace=0.15,
            left=0.105,
            right=0.985,
            top=0.96,
            bottom=0.078,
        )
        axa = fig.add_subplot(gs[0])
        strip = fig.add_subplot(gs[1], sharex=axa)
        axb = fig.add_subplot(gs[2], sharex=axa)
        for ax in (axa, axb):
            for a, b, ppm in ((0.0, sw, ppm_a), (sw, T_END, ppm_b)):
                if ppm > 0:
                    ax.axvspan(a, b, color=SHADE_H2, lw=0, zorder=0)
            ax.axvline(sw, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axa.tick_params(axis="x", labelbottom=False)
        strip.tick_params(axis="x", labelbottom=False)
        for y, wins, color in (
            (1.0, experimental_data.irradiation_segments(run), IRR),
            (0.0, beam_blocks(run), CYC),
        ):
            for a, b in wins:
                w = max(b - a, 0.006 * T_END)
                strip.add_patch(
                    plt.Rectangle(
                        (0.5 * (a + b) - 0.5 * w, y - 0.3),
                        w,
                        0.6,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
        strip.set_ylim(-0.6, 1.6)
        strip.set_yticks([1.0, 0.0])
        strip.set_yticklabels(["Irradiation", "Cyclotron"], fontsize=FS_S, color=INK)
        strip.tick_params(axis="both", which="both", length=0)
        for sp in strip.spines.values():
            sp.set_visible(False)

        def panel(ax, items, total, t_meas, y_meas, ylabel, title, col, filled):
            """items bottom to top; each area named with its share, inside it or beside a leader line."""
            items = [it for it in items if it[1][-1] >= 0.05]
            ymax = max(total[-1], y_meas[-1])
            thin = {id(it[1]) for it in items if it[1][-1] < 0.13 * ymax}
            step = 10.0 if ymax > 30 else 5.0 if ymax > 15 else 2.0 if ymax > 5 else 1.0
            top = step * np.ceil((1.15 + 0.12 * len(thin)) * ymax / step)
            ax.set_ylim(0, top)
            ax.set_yticks(np.arange(0, top + 0.5 * step, step))
            ax.set_ylabel(ylabel)
            ax.hlines(
                np.arange(step, top + 0.5 * step, step),
                0,
                T_END,
                color=GRID,
                lw=0.6,
                zorder=0.5,
            )
            base, n_thin = np.zeros_like(total), 0
            for name, y, color in items:
                ax.fill_between(
                    t, base, base + y, color=color, alpha=0.4, lw=0, zorder=2
                )
                share = f"{name}  {100 * y[-1] / total[-1]:.0f} %"
                if (
                    id(y) in thin
                ):  # labels stacked at the upper right, leaders that do not cross
                    xa = (0.955 - 0.06 * n_thin) * T_END
                    yt = top * (0.87 - 0.085 * n_thin)
                    n_thin += 1
                    ax.annotate(
                        share,
                        xy=(xa, np.interp(xa, t, base + 0.5 * y)),
                        xytext=((0.975 - 0.075 * (n_thin - 1)) * T_END, yt),
                        fontsize=FS_S,
                        color=INK,
                        ha="right",
                        va="center",
                        arrowprops=dict(
                            arrowstyle="-",
                            color=INK2,
                            lw=0.5,
                            shrinkA=2,
                            shrinkB=0,
                            relpos=(0.97, 0.0),
                        ),
                        zorder=7,
                    )
                else:
                    ax.text(
                        0.975 * T_END,
                        np.interp(0.9 * T_END, t, base + 0.5 * y),
                        share,
                        fontsize=FS_S,
                        ha="right",
                        va="center",
                        color=INK,
                        zorder=7,
                    )
                base = base + y
                ax.plot(t, base, color="white", lw=0.6, zorder=3)
            mk = dict(
                marker="o",
                ls="none",
                ms=4.6,
                mfc=col if filled else "white",
                mec="white" if filled else col,
                mew=0.6 if filled else 1.2,
            )
            ax.plot(t, total, color=col, lw=1.8, zorder=4)
            ax.plot(t_meas, y_meas, zorder=5, clip_on=False, **mk)
            ax.set_title(title, fontsize=FS + 0.5, color=INK, loc="center", pad=4)
            ax.legend(
                handles=[
                    Line2D([], [], color=col, lw=1.8, label="Model"),
                    Line2D([], [], label="Measured", **mk),
                ],
                loc="upper left",
                frameon=False,
                fontsize=FS_S,
                handlelength=1.5,
                labelspacing=0.3,
                borderaxespad=0.5,
            )
            return items

        ia = panel(
            axa,
            [
                ("Sparge, direct", iv_sp, C_SP),
                ("Free surface, direct", iv_fs, C_FS),
                ("IV holdup, this run", iv_pool_own, C_POOL),
                (f"IV holdup inherited from Run {prev}", iv_pool_inh, C_POOL_INH),
            ],
            iv_model,
            c["iv_t"],
            c["iv"],
            "Cumulative IV recovery (Bq)",
            f"(a) Run {run}: IV recovery by origin",
            IVC,
            True,
        )
        ib = panel(
            axb,
            [
                (f"OV holdup inherited from Run {prev}", ov_pool_inh, C_POOL_INH),
                (f"Stored in the wall before Run {run}", ov_wall_inh, C_WALL_INH),
                (f"Bred in Run {run}", ov_bred, C_BRED),
            ],
            ov_model,
            c["ov_t"],
            c["ov"],
            "Cumulative OV recovery (Bq)",
            f"(b) Run {run}: OV recovery by origin",
            OVC,
            False,
        )
        for ax in (axa, axb):
            tr = ax.get_xaxis_transform()
            ax.text(
                sw - 0.012 * T_END,
                0.975,
                GAS[ppm_a],
                transform=tr,
                ha="right",
                va="top",
                fontsize=FS_S,
                color=INK2,
            )
            ax.text(
                sw + 0.012 * T_END,
                0.975,
                GAS[ppm_b],
                transform=tr,
                ha="left",
                va="top",
                fontsize=FS_S,
                color=INK2,
            )
        axb.set_xlim(0, T_END * 1.012)
        axb.set_xticks(np.arange(0, T_END + 0.1, 5))
        axb.set_xlabel(f"Time in Run {run} (days)")
        fig.align_ylabels([axa, axb])
        save(fig, f"fig6_pathways_run{run}")
    for lab, items, tot, meas in (
        ("IV", ia, iv_model, c["iv"]),
        ("OV", ib, ov_model, c["ov"]),
    ):
        print(
            f"   {lab}: model {tot[-1]:.2f} = "
            + " + ".join(f"{y[-1]:.2f} [{n}]" for n, y, _ in items)
            + f"; measured {float(meas[-1]):.2f}"
        )


# ================================================================== Fig. 7: parameter ranges
LABELS = {
    "tbr": "TBR",
    "rho_1": "$\\rho_1$",
    "rho_2": "$\\rho_2$",
    "rho_3": "$\\rho_3$",
    "rho_12": "$\\rho_{12}$",
    "eta": "$\\eta$ [atoms/C]",
    "tau_iv": "$\\tau_\\mathrm{IV}$ [s]",
    "Bi": "Bi",
    "lam_h2": "$\\lambda_{\\mathrm{H}_2}$ [1/s]",
    "lam_he": "$\\lambda_\\mathrm{He}$ [1/s]",
    "L_he": "$L_\\mathrm{He}$ [m$^{-2}$s$^{-1}$Pa$^{-1}$]",
    "L_h2": "$L_{1000}$ [m$^{-2}$s$^{-1}$Pa$^{-1}$]",
    "alpha_iv": "$\\alpha_\\mathrm{IV}$",
    "alpha_ov": "$\\alpha_\\mathrm{OV}$",
    "k_iv": "$k_\\mathrm{IV}$ [1/s]",
    "k_ov": "$k_\\mathrm{OV}$ [1/s]",
    "k_ov_he": "$k_\\mathrm{OV,He}$ [1/s]",
}


def fig7_parameter_ranges():
    base = parameters.read_params(os.path.join(W, "calibrated_surrogate.txt"))
    with open(os.path.join(W, "festim", "admissibility_threshold.txt")) as fh:
        thr = float(fh.read())
    pts = []
    with open(os.path.join(W, "parameter_ranges", "profile_points.jsonl")) as fh:
        for line in fh:
            try:
                pts.append(json.loads(line))
            except json.JSONDecodeError:  # half-written last line of an interrupted run
                pass
    post0 = likelihood.objective(base)
    best = min([post0] + [p["post"] for p in pts])
    names = [n for n, *_ in parameters.SPACE]
    ncol = 6
    fig, axes = plt.subplots((len(names) + ncol - 1) // ncol, ncol, figsize=(17, 8.6))
    for ax, n in zip(axes.flat, names):
        xs = np.array([base[n]] + [p["value"] for p in pts if p["param"] == n])
        ys = np.array([post0] + [p["post"] for p in pts if p["param"] == n]) - best
        o = np.argsort(xs)
        label = LABELS.get(n, n)
        # log axis only for ranges over a decade
        log_axis = parameters.SPEC[n][5] and xs.max() > 20 * xs.min()
        scale = 1.0
        if (
            not log_axis and not 1e-2 <= np.abs(xs).max() < 1e4
        ):  # power of ten into the axis label
            e = int(np.floor(np.log10(np.abs(xs).max())))
            scale = 10.0**e
            label = (
                label.replace("[", f"[$10^{{{e}}}$ ")
                if "[" in label
                else f"{label} [$10^{{{e}}}$]"
            )
        ax.axhspan(0, thr, color="#E4F3EE", lw=0, zorder=0)
        ax.axhline(thr, color=SWC, ls=(0, (5, 3)), lw=1.2, zorder=1)
        ax.plot(xs[o] / scale, ys[o], "-o", color="#2a78d6", ms=3.5, lw=1.4, zorder=3)
        ax.plot(base[n] / scale, post0 - best, "o", color="#0b0b0b", ms=6, zorder=4)
        if log_axis:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_ylim(-0.3, 2.2 * thr)
        ax.set_xlabel(label, fontsize=12)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.18)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("$\\Delta$(−log posterior)", fontsize=11)
    for ax in list(axes.flat)[len(names) :]:
        ax.axis("off")
    fig.legend(
        [
            Line2D([], [], color="#2a78d6", marker="o", ms=3.5, lw=1.4),
            Line2D([], [], color="#0b0b0b", marker="o", ms=6, ls="none"),
            Line2D([], [], color=SWC, ls=(0, (5, 3)), lw=1.2),
        ],
        [
            "profile (all other parameters re-optimised)",
            "calibrated value",
            f"admissibility threshold 1.92 $\\chi^2_\\mathrm{{red}}$ = {thr:.2f}",
        ],
        loc="upper center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(
        top=0.93, bottom=0.08, left=0.05, right=0.99, hspace=0.55, wspace=0.32
    )
    save(fig, "fig7_parameter_ranges")


if __name__ == "__main__":
    fig1_geometry()
    fig2_mechanism()
    load_model_output()
    fig3_training()
    fig4_prediction()
    fig5_inventory()
    fig6_pathways(3)
    fig7_parameter_ranges()
    _nll, tr, hd = likelihood.scores(P)
    print(f"FESTIM training -logL {tr:.2f}, held-out {hd:.2f}")
