"""
Clean Fig 3 (comoving clock rate) and Fig 5 (finite-size comparison)
restricted to the Lorentz validity regime gamma <= 5, with both
the Lorentz prediction m/gamma and the lattice prediction
omega_lat = epsilon_floq - p * v_g_floq plotted.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

with open(DATA / "10_band_resolved_clock.json") as f:
    d = json.load(f)

# ---- Fig 3 (clean, gamma <= 5) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

m = d["runs"][0]["m_floq"]
gammas = np.linspace(1, 5.0, 300)
ax1.plot(gammas, m / gammas, 'k--', label=r'Lorentz: $m/\gamma$', lw=1.6)
ax1.axhline(m, color='grey', ls=':', alpha=0.7, label=fr'$m={m:.4f}$ (rest)')

# main result: L=128 measured (band-resolved); restrict to gamma <= 5
run = d["runs"][0]   # L=128
g = np.array([r["gamma_floq"] for r in run["results"]])
om_meas = np.array([r["omega_comoving_meas"] for r in run["results"]])
m_over_g = np.array([r["m_over_gamma_pred"] for r in run["results"]])
mask = g <= 5.0
ax1.plot(g[mask], om_meas[mask], 'o', color='crimson', ms=8,
         label=r'measured $\omega_{\rm comov}$ ($L=128$)', zorder=3)
ax1.plot(g[mask], m_over_g[mask], '.', color='black', ms=4, alpha=0.7,
         label=r'$m/\gamma$ from Floquet $E$')

ax1.set_xlabel(r'$\gamma = E/m$')
ax1.set_ylabel(r'comoving phase rate (lab time units)')
ax1.set_title(r'Operational kinematic clock rate vs Lorentz prediction')
ax1.legend(loc='upper right', framealpha=0.95)
ax1.grid(alpha=0.3)
ax1.set_xlim(0.95, 5.05)
ax1.set_ylim(0, 0.10)

# Right panel: relative deviation, shaded validity regime
ax2.axhline(0, color='k', lw=0.7)
ax2.axhspan(-1, 1, color='green', alpha=0.18, label='Lorentz regime (|err|<1%)')
ax2.axhspan(-10, 10, color='gold', alpha=0.10)
err_pct = np.array([100*r["rel_err"] for r in run["results"]])
ax2.plot(g[mask], err_pct[mask], 'o-', color='crimson', ms=7,
         label='measured $-$ Lorentz, relative')
ax2.set_xlabel(r'$\gamma = E/m$')
ax2.set_ylabel(r'relative deviation (%)')
ax2.set_title(r'Departure from Lorentz form within $\gamma \leq 5$')
ax2.legend(loc='lower left')
ax2.grid(alpha=0.3)
ax2.set_xlim(0.95, 5.05)
ax2.set_ylim(-25, 5)

plt.tight_layout()
out_fig3 = FIG / "fig3_comoving_clock_rate_clean.pdf"
out_fig3_png = FIG / "fig3_comoving_clock_rate_clean.png"
plt.savefig(out_fig3, bbox_inches='tight')
plt.savefig(out_fig3_png, dpi=150, bbox_inches='tight')
print(f"saved {out_fig3}")
plt.close()

# ---- Fig 5 (finite-size, both restricted to gamma <= 5) ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

ax1.plot(gammas, m / gammas, 'k--', label=r'Lorentz: $m/\gamma$', lw=1.6)
ax1.axhline(m, color='grey', ls=':', alpha=0.6)

colors = {128: 'crimson', 256: 'royalblue'}
markers = {128: 'o', 256: 's'}
for run in d["runs"]:
    L = run["L"]; sigma = run["sigma"]
    g = np.array([r["gamma_floq"] for r in run["results"]])
    om = np.array([r["omega_comoving_meas"] for r in run["results"]])
    mask = g <= 5.0
    ax1.plot(g[mask], om[mask], marker=markers[L], color=colors[L], ls='',
             label=f'$L={L}$, $\\sigma={int(sigma)}$', ms=8, alpha=0.85)

ax1.set_xlabel(r'$\gamma = E/m$')
ax1.set_ylabel(r'comoving phase rate (lab time units)')
ax1.set_title(r'$\omega_{\rm comov}$: $L=128$ vs $L=256$ (Lorentz regime)')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_xlim(0.95, 5.05)
ax1.set_ylim(0, 0.10)

# Right panel: relative deviation, both runs
ax2.axhline(0, color='k', lw=0.7)
ax2.axhspan(-1, 1, color='green', alpha=0.18, label='$\\pm$1%')
ax2.axhspan(-10, 10, color='gold', alpha=0.10, label='$\\pm$10%')
for run in d["runs"]:
    L = run["L"]
    g = np.array([r["gamma_floq"] for r in run["results"]])
    err = np.array([100*r["rel_err"] for r in run["results"]])
    mask = g <= 5.0
    ax2.plot(g[mask], err[mask], marker=markers[L], color=colors[L], ls='-',
             label=f'$L={L}$', ms=7, alpha=0.9)
ax2.set_xlabel(r'$\gamma = E/m$')
ax2.set_ylabel(r'relative deviation $(\omega_{\rm comov}-m/\gamma)/(m/\gamma)$ (%)')
ax2.set_title(r'Convergence with box size in the Lorentz regime')
ax2.legend(loc='lower left')
ax2.grid(alpha=0.3)
ax2.set_xlim(0.95, 5.05)
ax2.set_ylim(-25, 5)

plt.tight_layout()
out_fig5 = FIG / "fig5_finite_size_comparison.pdf"
out_fig5_png = FIG / "fig5_finite_size_comparison.png"
plt.savefig(out_fig5, bbox_inches='tight')
plt.savefig(out_fig5_png, dpi=150, bbox_inches='tight')
print(f"saved {out_fig5}")
plt.close()
