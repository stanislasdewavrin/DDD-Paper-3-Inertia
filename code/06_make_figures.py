"""Paper III — figure generation.
Produces four figures from existing data + one small wavepacket run.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0 / (2 * PI)
T_PERIOD = 2.0
M_FLOQ = LAMBDA / T_PERIOD  # 1/(4 pi)
C_EFF = 1.0 / T_PERIOD


# ============================================================
# Helpers (copied from earlier scripts)
# ============================================================
def step_A(psi, tau_A, L):
    a_k = np.fft.fft2(psi[..., 0]); b_k = np.fft.fft2(psi[..., 1])
    kx = 2*PI*np.fft.fftfreq(L); ky = 2*PI*np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dx = np.sin(KX); dy = np.sin(KY); dz = np.cos(KX)+np.cos(KY)
    dn = np.sqrt(dx**2+dy**2+dz**2+1e-15)
    c = np.cos(dn*tau_A); s = np.sin(dn*tau_A)
    U00 = c - 1j*s*dz/dn
    U01 = -1j*s*(dx-1j*dy)/dn
    U10 = -1j*s*(dx+1j*dy)/dn
    U11 = c + 1j*s*dz/dn
    return np.stack([np.fft.ifft2(U00*a_k+U01*b_k),
                     np.fft.ifft2(U10*a_k+U11*b_k)], axis=-1)


def step_B(psi, tau_B, L, lam=LAMBDA):
    a_k = np.fft.fft2(psi[..., 0]); b_k = np.fft.fft2(psi[..., 1])
    kx = 2*PI*np.fft.fftfreq(L); ky = 2*PI*np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dz = lam*np.cos(KX+KY)
    ph = np.exp(-1j*dz*tau_B)
    return np.stack([np.fft.ifft2(ph*a_k),
                     np.fft.ifft2(np.conj(ph)*b_k)], axis=-1)


def floquet_diag(kx, ky, tau_A=1.0, tau_B=1.0, lam=LAMBDA):
    dxA = np.sin(kx); dyA = np.sin(ky); dzA = np.cos(kx)+np.cos(ky)
    nA = np.sqrt(dxA**2+dyA**2+dzA**2+1e-15)
    cA = np.cos(nA*tau_A); sA = np.sin(nA*tau_A)
    UA = np.array([[cA-1j*sA*dzA/nA, -1j*sA*(dxA-1j*dyA)/nA],
                   [-1j*sA*(dxA+1j*dyA)/nA, cA+1j*sA*dzA/nA]], dtype=complex)
    dzB = lam*np.cos(kx+ky)
    UB = np.array([[np.exp(-1j*dzB*tau_B), 0],
                   [0, np.exp(1j*dzB*tau_B)]], dtype=complex)
    UF = UB @ UA
    evals, evecs = np.linalg.eig(UF)
    quasi = -np.angle(evals) / (tau_A + tau_B)
    if quasi[0] < quasi[1]:
        return float(quasi[0]), float(quasi[1]), evecs[:, 0]
    return float(quasi[1]), float(quasi[0]), evecs[:, 1]


def make_wp(L, k0, sigma):
    eps_lo, _, u_lo = floquet_diag(k0[0], k0[1])
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    env = np.exp(-((xs-L/2)**2 + (ys-L/2)**2) / (2*sigma**2))
    plane = np.exp(1j*(k0[0]*xs + k0[1]*ys))
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane * u_lo[0]
    psi[..., 1] = env * plane * u_lo[1]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi


# ============================================================
# Figure 1: Wavepacket propagation snapshots
# ============================================================
print("Figure 1: wavepacket propagation snapshots...")
L = 96; sigma = 7.0
delta = 0.4  # gives v_g ≈ 0.41
k0 = (PI - delta, 0.0)
psi = make_wp(L, k0, sigma)

snapshots = {0: np.abs(psi[..., 0])**2 + np.abs(psi[..., 1])**2}
for it in range(1, 81):
    psi = step_A(psi, 1.0, L); psi = step_B(psi, 1.0, L)
    if it in (10, 25, 50, 80):
        snapshots[it] = np.abs(psi[..., 0])**2 + np.abs(psi[..., 1])**2

fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), constrained_layout=True)
vmax = max(s.max() for s in snapshots.values())
for ax, (t, rho) in zip(axes, sorted(snapshots.items())):
    im = ax.imshow(rho.T, origin='lower', cmap='viridis', vmin=0, vmax=vmax,
                   extent=[0, L, 0, L])
    ax.set_title(rf"$t = {t}T$ (lab time {2*t})", fontsize=10)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$" if t == 0 else "")
    ax.axhline(L/2, color='white', lw=0.4, alpha=0.5)
fig.suptitle(rf"Wavepacket at $k_0 = (\pi - {delta:.1f}, 0)$, "
             rf"propagating at $v_g \approx {0.41:.2f}\,c_{{\rm eff}}$ "
             rf"on $L = {L}$ lattice. Density $|\psi|^2$.",
             fontsize=10, y=1.05)
fig.savefig(FIG / "fig1_wavepacket_propagation.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig1_wavepacket_propagation.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 2: Floquet dispersion + Lorentz fit (along k_x at k_y=0)
# ============================================================
print("Figure 2: Floquet dispersion...")
kxs = np.linspace(0, 2*PI, 401)
eps_lo = np.array([floquet_diag(kx, 0.0)[0] for kx in kxs])
eps_hi = -eps_lo

# Lorentz prediction near (pi, 0)
deltas_fine = np.linspace(-0.7, 0.7, 200)
E_lor = np.sqrt(M_FLOQ**2 + (deltas_fine * C_EFF)**2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

ax = axes[0]
ax.plot(kxs, eps_hi, 'b-', lw=1.5, label=r'$\varepsilon_+$ (Floquet)')
ax.plot(kxs, eps_lo, 'r-', lw=1.5, label=r'$\varepsilon_-$ (Floquet)')
ax.axhspan(-M_FLOQ, M_FLOQ, alpha=0.15, color='gray', label='gap')
ax.axvline(PI, color='k', ls=':', alpha=0.4)
ax.set_xlabel(r'$k_x$ (with $k_y = 0$)')
ax.set_ylabel(r'Floquet quasi-energy $\varepsilon$')
ax.set_title(r'Full quasi-energy band structure')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks([0, PI/2, PI, 3*PI/2, 2*PI])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

ax = axes[1]
mask = (kxs > PI - 0.7) & (kxs < PI + 0.7)
ax.plot(kxs[mask] - PI, np.abs(eps_lo[mask]), 'ro', ms=3, label=r'$|\varepsilon_-|$ Floquet')
ax.plot(deltas_fine, E_lor, 'k--', lw=1.5,
        label=rf'Lorentz $\sqrt{{m^2 + (c_{{\rm eff}}\,p)^2}}$')
ax.axhline(M_FLOQ, color='gray', ls=':', alpha=0.6, label=rf'$m = 1/(4\pi)$')
ax.set_xlabel(r'$\delta = k_x - \pi$ (offset from gap)')
ax.set_ylabel(r'$|\varepsilon_-|$')
ax.set_title(r'Zoom on the gap at $(\pi, 0)$ with Lorentz fit')
ax.legend(loc='upper center', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(FIG / "fig2_floquet_dispersion.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig2_floquet_dispersion.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# ============================================================
# Figure 3 + 4: Comoving clock rate result + lattice deviation
# ============================================================
print("Figures 3 + 4: comoving clock rate + lattice deviation...")
with open(DATA / "05_comoving_clock.json") as f:
    d = json.load(f)
res = d["results"]
deltas = np.array([r["delta"] for r in res])
E = np.array([r["E_floq"] for r in res])
gamma = np.array([r["gamma_floq"] for r in res])
m_over_gamma_pred = np.array([r["m_over_gamma_pred"] for r in res])
omega_meas = np.array([r["omega_comoving_meas"] for r in res])
rel_err = np.array([r["rel_err"] for r in res])
m = d["m_floq"]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
gamma_smooth = np.linspace(1, gamma.max(), 200)
ax.plot(gamma_smooth, m / gamma_smooth, 'k--', lw=1.5,
        label=r'Lorentz: $m/\gamma$')
ax.plot(gamma, m_over_gamma_pred, 'k.', ms=4, alpha=0.5,
        label=r'$m/\gamma$ from Floquet $E$')
ax.plot(gamma, omega_meas, 'ro', ms=8, label=r'measured $\omega_{\rm comov}$')
ax.axhline(m, color='gray', ls=':', alpha=0.5, label=rf'$m = 1/(4\pi)$')
ax.set_xlabel(r'$\gamma = E/m$')
ax.set_ylabel(r'comoving phase rate (lab time units)')
ax.set_title('Operational kinematic clock rate vs Lorentz prediction')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0.95, gamma.max() * 1.05)
fig.tight_layout()
fig.savefig(FIG / "fig3_comoving_clock_rate.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig3_comoving_clock_rate.png", dpi=150, bbox_inches='tight')
plt.close(fig)


# Lattice deviation
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.axhline(0, color='k', ls=':', alpha=0.5)
ax.plot(gamma, 100 * rel_err, 'rs-', ms=7, lw=1.5)
ax.fill_between([0, 5], -1, 1, color='green', alpha=0.1,
                label=r'$|err| < 1\%$ (Lorentz regime)')
ax.fill_between([0, 5], -10, 10, color='gold', alpha=0.1,
                label=r'$|err| < 10\%$ (mild lattice)')
ax.set_xlabel(r'$\gamma = E/m$')
ax.set_ylabel(r'$(\omega_{\rm meas} - m/\gamma)\,/\,(m/\gamma)$  [%]')
ax.set_title('Lattice corrections to the kinematic clock-rate')
ax.set_xlim(0.95, gamma.max() * 1.05)
ax.set_ylim(-25, 5)
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig4_lattice_deviation.pdf", bbox_inches='tight')
fig.savefig(FIG / "fig4_lattice_deviation.png", dpi=150, bbox_inches='tight')
plt.close(fig)


print("All figures saved to:", FIG)
