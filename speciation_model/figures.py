"""Figures of the speciation model, all from a FESTIM chain with the collection lines on top.

  fig1_mechanism     every state and step of the model on the vessel, with the rates of the parameter set
  fig2_recovery      IV and OV cumulative recovery (insoluble + soluble), six runs
  fig3_forms         the same split into the insoluble and the soluble form, with the parts of each curve

    python figures.py PARAMS CHAIN OUT_DIR
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import model  # noqa: E402  (puts the repository on the path)
import experimental_data  # noqa: E402
import festim_model  # noqa: E402

PARAMS, CHAIN, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
th = model.read_params(PARAMS)
sim = festim_model.campaign(th, CHAIN)
BQ = model.BQ
ORDER, TAG = (1, 2, 3, 12, 4, 6), "abcdef"
IVC, OVC, IRR, SWC = "#1B9E77", "#D9601F", "#E8709A", "#8A8A8A"
C_TF, C_HT, C_T, C_MEM, INK, MUTED = "#B5361C", "#1f5fae", "#1B7A50", "#6B4FA0", "#222222", "#6A6A6A"
plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name, dpi=170):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(os.path.join(OUT, f"{name}.png"))


def g(v):
    return f"{v:.2g}".replace("e-0", "e-")


# ------------------------------------------------------------------ fig 1: mechanism
fig = plt.figure(figsize=(15.5, 9.2))
ax = fig.add_axes([0.0, 0.0, 0.60, 1.0])
ax.set_xlim(0, 10.2)
ax.set_ylim(-0.9, 7.6)
ax.set_aspect("equal")
ax.axis("off")


def arrow(p0, p1, c, lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color=c, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=6))


def num(x, y, n, c):
    ax.add_patch(Circle((x, y), 0.25, facecolor=c, edgecolor="white", lw=1.0, zorder=12))
    ax.text(x, y - 0.01, str(n), ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=13)


def chip(x, y, text, c, fs=10.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color="white", zorder=14,
            bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="white", lw=1.0))


def box(x0, y0, w, h, text, ec, fc="white", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor=fc,
                                edgecolor=ec, lw=1.8, zorder=5))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs, color=ec,
            fontweight="bold" if bold else "normal", zorder=6)


EX0, EX1, EY0, EY1 = 0.25, 5.95, 0.25, 5.35
VX0, VX1, VY0, VY1 = 0.75, 5.45, 0.45, 4.95
IX0, IX1, YM0, YFS, YCAP = 1.05, 4.85, 0.95, 2.95, 4.65
ax.add_patch(Rectangle((EX0, EY0), EX1 - EX0, EY1 - EY0, fc="#E7F0E7", ec="#9DBE9D", lw=1.5, zorder=1))
ax.add_patch(Rectangle((VX0, VY0), VX1 - VX0, VY1 - VY0, fc="#7FA6CE", ec=OVC, lw=2.6, zorder=2))
ax.add_patch(Rectangle((IX0, YFS), IX1 - IX0, YCAP - YFS, fc="#E9F2F7", ec="#8C8C8C", lw=2.0, zorder=3))
ax.add_patch(Rectangle((IX0, YM0), IX1 - IX0, YFS - YM0, fc="#F2B24A", ec="none", zorder=3))
ax.plot([IX0, IX0, IX1, IX1], [YFS, YM0, YM0, YFS], color="#B22222", lw=2.4, zorder=4)
ax.plot([IX0, IX1], [YFS, YFS], color=IVC, lw=2.6, zorder=4)
SPX = 1.6
ax.add_patch(Rectangle((SPX - 0.07, YM0 + 0.25), 0.14, VY1 + 0.3 - YM0 - 0.25, fc="#DCDCDC", ec="#B0B0B0", lw=1, zorder=4))
for bx, by in ((1.45, 1.40), (1.78, 1.70), (1.47, 2.05), (1.76, 2.40), (1.52, 2.75)):
    ax.add_patch(Circle((bx, by), 0.075, fc="white", ec="#7A9CC0", lw=1.0, zorder=5))
ax.text(VX0 + 0.12, VY1 - 0.22, "Inconel 625", fontsize=11, fontweight="bold", color="#2B4C7E", zorder=6)
ax.text(2.05, YCAP - 0.3, "headspace (swept, inert wall)", fontsize=9.5, color="#6F8798", zorder=6)
ax.text(2.75, YM0 + 0.1, "ClLiF melt", fontsize=11.5, fontweight="bold", color="#6B4A10", zorder=6)
ax.text(EX0 + 0.12, EY0 - 0.02, "OV enclosure (swept)", fontsize=9.5, color="#8B4513", zorder=6, va="bottom")
ax.text(SPX + 0.13, VY1 + 0.15, "sparger", fontsize=9, color=MUTED, zorder=6)
TFp, HTp = (2.55, 2.2), (3.45, 1.55)
chip(*TFp, "TF", C_TF)
chip(*HTp, "HT", C_HT)
num(TFp[0], TFp[1] + 0.45, 1, "#6B4A10")
arrow((2.82, 2.05), (3.2, 1.72), C_TF, lw=2.2)
num(2.72, 1.62, 2, C_TF)
arrow((3.75, 1.45), (IX1 - 0.27, 1.45), C_HT, lw=1.9)
num(IX1, 1.45, 3, C_HT)
chip(5.15, 2.85, "T", C_T, fs=9.5)
num(5.15, 1.9, 6, C_T)
for y0, y1 in ((1.45, 1.63), (2.17, 2.66)):
    ax.plot([5.15, 5.15], [y0, y1], color=C_T, lw=1.4, ls=(0, (1.5, 1.5)), zorder=7)
ax.plot([IX1 + 0.27, 5.15], [1.45, 1.45], color=C_T, lw=1.4, ls=(0, (1.5, 1.5)), zorder=7)
IVY, OVY, LX = 6.05, 2.85, 6.4
arrow((3.45, 1.75), (3.45, YFS - 0.27), C_HT, lw=2.0)
ax.plot([3.45, 3.45], [YFS + 0.25, IVY], color=IVC, lw=2.2, zorder=6)
num(3.45, YFS, 4, IVC)
ax.plot([SPX, SPX, 3.45], [VY1 + 0.3, IVY, IVY], color=IVC, lw=2.2, ls=(0, (4, 2)), zorder=6)
num(SPX + 0.42, 1.2, 5, IVC)
ax.plot([3.45, LX], [IVY, IVY], color=IVC, lw=2.2, zorder=6)
arrow((5.32, OVY), (VX1 + 0.03, OVY), C_T, lw=1.6)
ax.plot([VX1 + 0.57, LX], [OVY, OVY], color=OVC, lw=2.2, zorder=6)
num(VX1 + 0.31, OVY, 7, OVC)
box(LX + 0.2, IVY - 1.15, 1.65, 0.65, "IV-line holdup", IVC)
box(LX + 0.2, OVY - 1.15, 1.65, 0.65, "OV-line holdup", OVC)
box(8.45, IVY - 0.38, 1.7, 0.76, "IV bubbler\ninsoluble | soluble", IVC, fc="#E3F2EC", fs=9.5, bold=True)
box(8.45, OVY - 0.38, 1.7, 0.76, "OV bubbler\ninsoluble | soluble", OVC, fc="#FBE8DD", fs=9.5, bold=True)
arrow((LX, IVY), (8.45, IVY), IVC)
arrow((LX, OVY), (8.45, OVY), OVC)
for y, c, n in ((IVY, IVC, 8), (OVY, OVC, 9)):
    ax.plot([LX, LX, LX + 0.2], [y, y - 0.82, y - 0.82], color=c, lw=2.0, zorder=6)
    ax.plot([LX + 1.85, 8.32, 8.32], [y - 0.82, y - 0.82, y - 0.1], color=c, lw=2.0, zorder=6)
    arrow((8.32, y - 0.12), (8.32, y - 0.02), c, lw=2.0)
    num(LX, y - 0.45, n, c)
MY = -0.55
for xs, ys in (([3.0, 3.0], [EY0, MY]), ([LX + 1.03, LX + 1.03], [OVY - 1.15, MY]),
               ([LX + 1.03, LX + 1.03], [IVY - 1.15, IVY - 1.5]), ([3.0, 9.7], [MY, MY])):
    ax.plot(xs, ys, color=C_MEM, lw=1.8, ls=(0, (2.5, 1.5)), zorder=6)
ax.text(LX + 1.03, IVY - 1.62, "to 10", fontsize=8.5, color=C_MEM, ha="center", va="top")
arrow((9.7, MY), (10.15, MY), C_MEM, lw=1.8)
ax.text(9.7, MY + 0.2, "next run", fontsize=10, color=C_MEM, ha="center")
num(5.0, MY, 10, C_MEM)
ax.text(3.1, 7.25, "where the tritium is, and what it turns into", fontsize=13, fontweight="bold", color=INK, ha="center")
ax.text(3.1, 6.85, "hot zone (FESTIM)", fontsize=10.5, style="italic", color=MUTED, ha="center")
ax.text(7.6, 6.85, "collection system (0-D)", fontsize=10.5, style="italic", color=MUTED, ha="center")
for x0, lab, c, txt in ((0.45, "TF", C_TF, "oxidised, in the melt"), (2.55, "HT", C_HT, "molecular, in the melt"),
                        (5.55, "T", C_T, "atom, in the Inconel")):
    chip(x0, 6.45, lab, c, fs=9.5)
    ax.text(x0 + 0.3, 6.45, txt, fontsize=8.5, color=INK, va="center")

r2 = sim[2]
mem = dict(melt=(r2["hz"]["TF"][-1] + r2["hz"]["HT"][-1]) * BQ, wall=r2["hz"]["wall"][-1] * BQ,
           iv=r2["pools"][0] * BQ, ov=r2["pools"][1] * BQ)
lam3 = model.lambda_henry(model.temperature(3))
TAGS = {"fit": ("fitted", "#1B7A50"), "lit": ("literature", "#4A6FA5"), "ass": ("assumption", MUTED),
        "out": ("model output", C_MEM)}
rows = [
    (1, "#6B4A10", r"$^6$Li(n,$\alpha$)T $\rightarrow$ TF", "all tritium is born oxidised, in the melt", "ass"),
    (2, C_TF, r"TF $\rightarrow$ HT in the melt",
     f"{g(th['k_red'])} /s  + {g(th['k_sp'])} /s if sparged  + {g(th['k_x'])}$\\,(x/1000)^{{{th['n_x']:.2f}}}$ /s with H$_2$", "fit"),
    (3, C_HT, r"HT(melt) $\leftrightarrow$ T(metal) + H",
     f"uptake {g(th['k_w'])} /s + {g(th['k_w_h2'])}$\\,(x/1000)$ /s;  return by detailed balance (literature $K_H$, $K_S$)", "fit"),
    (4, IVC, r"free surface $\rightarrow$ IV",
     f"HT {g(th['k_s'])} /s $\\rightarrow$ insoluble;  TF vapour {g(th['k_tf_s'])} /s $\\rightarrow$ soluble", "fit"),
    (5, IVC, r"sparge bubbles $\rightarrow$ IV",
     f"HT at {100 * th['e_sp']:.2g} % of Henry saturation = {g(th['e_sp'] * lam3)} /s $\\rightarrow$ insoluble;\n"
     f"TF {g(th['k_tf_sp'])} /s $\\rightarrow$ soluble", "fit"),
    (6, C_T, "T diffuses through the Inconel", "diffusivity and solubility from the literature", "lit"),
    (7, OVC, r"outer faces $\rightarrow$ OV",
     f"T+H$\\rightarrow$HT at $K_r c_H$ (literature; He background {g(th['x_bg'])} ppm H$_2$) $\\rightarrow$ insoluble;\n"
     f"T+OH$\\rightarrow$HTO at {g(th['k_hto'])} m/s $\\rightarrow$ soluble", "fit"),
    (8, IVC, "IV-line holdup", f"{100 * th['alpha_iv']:.1f} % of the insoluble stream held; released at "
     f"{g(th['k_iv_he'])} /s + {g(th['k_iv'])}$\\,(x/1000)$ /s", "fit"),
    (9, OVC, "OV-line holdup", f"{100 * th['alpha_ov']:.1f} % of the insoluble stream held; released at "
     f"{g(th['k_ov_he'])} /s + {g(th['k_ov'])}$\\,(x/1000)$ /s", "fit"),
    (10, C_MEM, "carried to the next run (end of run 2)",
     f"OV line {mem['ov']:.1f} Bq,  IV line {mem['iv']:.1f},  melt {mem['melt']:.1f},  wall {mem['wall']:.2f}", "out"),
]
tx = fig.add_axes([0.60, 0.0, 0.40, 1.0])
tx.axis("off")
tx.set_xlim(0, 1)
tx.set_ylim(0, 1)
y = 0.955
tx.text(0.0, y, "steps and rates", fontsize=12, fontweight="bold", color=INK)
y -= 0.05
for n, c, title, body, st in rows:
    tx.scatter([0.025], [y - 0.012], s=330, c=c, edgecolors="white", zorder=5)
    tx.text(0.025, y - 0.013, str(n), ha="center", va="center", fontsize=8.5, fontweight="bold", color="white", zorder=6)
    tx.text(0.06, y, title, fontsize=10.2, fontweight="bold", color=INK, va="top")
    tx.text(0.06, y - 0.03, body, fontsize=8.8, color=INK, va="top", linespacing=1.25)
    lab, sc = TAGS[st]
    tx.text(0.995, y, lab, fontsize=8, color=sc, va="top", ha="right", style="italic")
    y -= 0.067 + 0.022 * body.count("\n")
tx.text(0.0, 0.035, "x = H$_2$ in the sweep gas [ppm].  Fitted on runs 1, 2, 3, 12 (insoluble and soluble curves);\n"
        "source fixed at the transport model's calibration.  Soluble = water vials before the catalyst.",
        fontsize=8, color=MUTED, va="bottom")
save(fig, "fig1_mechanism")


# ------------------------------------------------------------------ fig 2, 3: recovery
def cond(r):
    sw, p0, p1 = experimental_data.H2_SCHEDULE[r]

    def gas(p):
        return "He" if p == 0 else f"{p / 1e4:g}% H$_2$"

    s = gas(p0) + ("" if sw is None else f" $\\rightarrow$ {gas(p1)}")
    return s + (", sparged" if experimental_data.HAS_SPARGER[r] else ", no sparger") + f", {experimental_data.RUN_TEMP_C[r]:.0f} °C"


def decorate(ax, r):
    for a, b in experimental_data.irradiation_segments(r):
        ax.axvspan(a, b, color=IRR, alpha=0.28, lw=0, zorder=0)
    sw = experimental_data.H2_SCHEDULE[r][0]
    if sw is not None:
        ax.axvline(sw, color=SWC, ls=(0, (5, 4)), lw=1.3, zorder=1)
    ax.grid(alpha=0.25)


def finish(ax, r):
    ax.relim()
    ax.autoscale_view()
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, experimental_data.t_end(r) * 1.02)


def soluble_as_measured(r, key):
    """The model's soluble curve as the clipped vials would read it (what the likelihood compares)."""
    te, _ = model.measured(r)[key]
    inc = np.diff(np.r_[0.0, np.interp(te, sim[r]["t"], sim[r]["curves"][key] * BQ)])
    return np.r_[0.0, te], np.r_[0.0, np.cumsum(model.soluble_as_measured(inc)[0])]


MK_F = dict(marker="o", ms=6.0, ls="none")
title = lambda k, r, extra="": f"({TAG[k]}{extra}) Run {r}" + ("" if r in model.TRAIN else "  — predicted")  # noqa: E731

fig, axs = plt.subplots(3, 2, figsize=(10.6, 12.0))
for k, r in enumerate(ORDER):
    ax = axs.flat[k]
    decorate(ax, r)
    meas = model.measured(r)
    for line, c, mfc in (("iv", IVC, IVC), ("ov", OVC, "white")):
        ax.plot(sim[r]["t"], (sim[r]["curves"][(line, "ins")] + sim[r]["curves"][(line, "sol")]) * BQ, color=c, lw=2.2, zorder=3)
        ax.plot(*meas[(line, "total")], mfc=mfc, mec=c, mew=1.6, **MK_F, zorder=4)
    finish(ax, r)
    ax.set_title(title(k, r), fontsize=13, fontweight="bold")
    ax.text(0.97, 0.47, cond(r), transform=ax.transAxes, fontsize=11, color=SWC, ha="right", va="center")
    ax.set_ylabel("cumulative recovery [Bq]", fontsize=13)
    ax.set_xlabel("time [day]", fontsize=13)
h = [Line2D([], [], mfc=IVC, mec=IVC, **MK_F), Line2D([], [], color=IVC, lw=2.2), Line2D([], [], mfc="white", mec=OVC, mew=1.6, **MK_F),
     Line2D([], [], color=OVC, lw=2.2), plt.Rectangle((0, 0), 1, 1, color=IRR, alpha=0.28, lw=0),
     Line2D([], [], color=SWC, ls=(0, (5, 4)), lw=1.3)]
fig.legend(h, ["IV measured", "IV model", "OV measured", "OV model", "irradiation", "gas switch"], fontsize=11.5, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=True, fancybox=False, edgecolor="#BDBDBD")
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2)
save(fig, "fig2_recovery")

PARTS = {("iv", "ins"): (("iv_ht_surface", "free surface"), ("iv_ht_bubbles", "bubbles"), ("iv_line", "IV line")),
         ("ov", "ins"): (("ov_ht", "outer faces, HT"), ("ov_line", "OV line")),
         ("iv", "sol"): (("iv_tf_surface", "TF, free surface"), ("iv_tf_bubbles", "TF, bubbles")),
         ("ov", "sol"): (("ov_hto", "outer faces, HTO"),)}
PART_LS = [(0, (5, 2)), (0, (1.2, 1.6)), (0, (6, 2, 1.2, 2))]
fig, axs = plt.subplots(6, 2, figsize=(12.6, 25.0))
for k, r in enumerate(ORDER):
    meas = model.measured(r)
    for j, form in enumerate(("ins", "sol")):
        ax = axs[k, j]
        decorate(ax, r)
        for line, c, mfc in (("iv", IVC, IVC), ("ov", OVC, "white")):
            key = (line, form)
            if form == "sol":
                ax.plot(*soluble_as_measured(r, key), color=c, lw=2.2, zorder=3)
                ax.plot(sim[r]["t"], sim[r]["curves"][key] * BQ, color=c, lw=1.2, alpha=0.55, zorder=2)
            else:
                ax.plot(sim[r]["t"], sim[r]["curves"][key] * BQ, color=c, lw=2.2, zorder=3)
                for (p, _lab), ls in zip(PARTS[key], PART_LS):
                    if sim[r]["parts"][p][-1] * BQ > 0.05:
                        ax.plot(sim[r]["t"], sim[r]["parts"][p] * BQ, color=c, lw=1.1, ls=ls, alpha=0.8, zorder=2)
            ax.plot(*meas[key], mfc=mfc, mec=c, mew=1.6, **MK_F, zorder=4)
        finish(ax, r)
        ax.set_ylabel(f"cumulative {'insoluble' if form == 'ins' else 'soluble'} [Bq]", fontsize=12)
        if k == len(ORDER) - 1:
            ax.set_xlabel("time [day]", fontsize=13)
        ax.set_title(title(k, r, j + 1) + (", insoluble" if form == "ins" else ", soluble"), fontsize=12, fontweight="bold")
    axs[k, 1].text(0.97, 0.12, cond(r), transform=axs[k, 1].transAxes, fontsize=10, color=SWC, ha="right")
h = [Line2D([], [], mfc=IVC, mec=IVC, **MK_F), Line2D([], [], mfc="white", mec=OVC, mew=1.6, **MK_F),
     Line2D([], [], color=IVC, lw=2.2), Line2D([], [], color=OVC, lw=2.2)]
l = ["IV measured", "OV measured", "IV model", "OV model"]
for ls, lab in zip(PART_LS, ("insoluble from the free surface / outer faces", "insoluble from the bubbles", "insoluble released by the line holdup")):
    h.append(Line2D([], [], color="#555555", lw=1.1, ls=ls))
    l.append(lab)
h.append(Line2D([], [], color="#555555", lw=1.2, alpha=0.55))
l.append(f"soluble released (thick line: as the vials read it, {model.VIAL_SIGMA:g} Bq noise per vial, negatives set to 0)")
fig.legend(h, l, fontsize=11, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.997), frameon=True, fancybox=False,
           edgecolor="#BDBDBD")
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.2)
save(fig, "fig3_forms", dpi=150)
