"""Plot finite-size comparison L=128 vs L=256 for omega_comov."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

with open(DATA / "08_finite_size_scan.json") as f:
    d = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {128: 'tab:red', 256: 'tab:blue'}
markers = {128: 'o', 256: 's'}

# Panel 1: omega_comov vs gamma
gammas = np.linspace(1, 8.5, 300)
m = d["runs"][0]["m_floq"]
ax1.plot(gammas, m / gammas, 'k--', label=r'Lorentz: $m/\gamma$', lw=1.5)
ax1.axhline(m, color='grey', ls=':', alpha=0.7, label=fr'$m={m:.4f}$ (rest)')

for run in d["runs"]:
    L = run["L"]
    g = [r["gamma_floq"] for r in run["results"]]
    om = [r["omega_comoving_meas"] for r in run["results"]]
    pred = [r["m_over_gamma_pred"] for r in run["results"]]
    ax1.plot(g, om, marker=markers[L], color=colors[L], ls='',
             label=f'$L={L}$ (measured)', ms=8, mfc=colors[L], alpha=0.85)
    ax1.plot(g, pred, '.', color=colors[L], ms=4, alpha=0.5)

ax1.set_xlabel(r'$\gamma = E/m$')
ax1.set_ylabel(r'comoving phase rate (lab time units)')
ax1.set_title(r'Operational $\omega_{\rm comov}$ vs $m/\gamma$ at two box sizes')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 0.10)

# Panel 2: relative error vs gamma
for run in d["runs"]:
    L = run["L"]
    g = [r["gamma_floq"] for r in run["results"]]
    err = [100*r["rel_err"] for r in run["results"]]
    ax2.plot(g, err, marker=markers[L], color=colors[L], ls='-',
             label=f'$L={L}$, $\\sigma=L/{int(L/24)}$ (={run["sigma"]:.0f})',
             ms=8, alpha=0.85)

ax2.axhline(0, color='k', lw=0.7)
ax2.axhspan(-1, 1, color='green', alpha=0.15, label='Lorentz regime ($\\pm$1%)')
ax2.axhspan(-10, 10, color='gold', alpha=0.10, label='Mild lattice ($\\pm$10%)')
ax2.set_xlabel(r'$\gamma = E/m$')
ax2.set_ylabel(r'relative deviation $(\omega_{\rm comov} - m/\gamma) / (m/\gamma)$ (%)')
ax2.set_title(r'Deviation from Lorentz: not a boundary effect')
ax2.legend(loc='upper left')
ax2.grid(alpha=0.3)
ax2.set_ylim(-30, 300)

plt.tight_layout()
out_pdf = FIG / "fig5_finite_size_comparison.pdf"
out_png = FIG / "fig5_finite_size_comparison.png"
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"saved {out_pdf} and {out_png}")
