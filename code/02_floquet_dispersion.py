"""
Paper III — experiment 02: Floquet dispersion + propagation check
==================================================================

Compute the actual Floquet quasi-energy of U_F = exp(-iH_B tau_B) exp(-iH_A tau_A)
per wavevector k, and verify that its gradient gives the wavepacket
group velocity.

For small tau the Floquet quasi-energy approaches H_A + H_B, but at
tau_A = tau_B = 1 (standard 2-step) Trotter corrections matter.
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0 / (2 * PI)


def floquet_quasienergy(kx, ky, tau_A=1.0, tau_B=1.0, lam=LAMBDA):
    """Quasi-energy at (kx,ky): eigenvalues of U_F = exp(-iH_B tau) exp(-iH_A tau).
    Returns (eps_minus, eps_plus) in [-pi/T, pi/T] with T = tau_A+tau_B."""
    # H_A = sin(kx) sx + sin(ky) sy + (cos kx + cos ky) sz
    dxA = np.sin(kx); dyA = np.sin(ky); dzA = np.cos(kx) + np.cos(ky)
    nA = np.sqrt(dxA**2 + dyA**2 + dzA**2 + 1e-15)
    cA = np.cos(nA * tau_A); sA = np.sin(nA * tau_A)
    UA = np.array([[cA - 1j*sA*dzA/nA, -1j*sA*(dxA - 1j*dyA)/nA],
                   [-1j*sA*(dxA + 1j*dyA)/nA, cA + 1j*sA*dzA/nA]], dtype=complex)
    # H_B = lam cos(kx+ky) sz
    dzB = lam * np.cos(kx + ky)
    UB = np.array([[np.exp(-1j*dzB*tau_B), 0],
                   [0, np.exp(1j*dzB*tau_B)]], dtype=complex)
    UF = UB @ UA
    evals = np.linalg.eigvals(UF)
    # quasi-energy = -arg(eigenvalue) / T
    T = tau_A + tau_B
    quasi = -np.angle(evals) / T
    return float(min(quasi)), float(max(quasi))


def floquet_quasienergy_grid(KX, KY, tau_A=1.0, tau_B=1.0):
    """Vectorised computation of quasi-energy on a grid."""
    # Compute U_F at each grid point and diagonalize.
    eps_lo = np.zeros_like(KX)
    eps_hi = np.zeros_like(KX)
    for i in range(KX.shape[0]):
        for j in range(KX.shape[1]):
            lo, hi = floquet_quasienergy(KX[i,j], KY[i,j], tau_A, tau_B)
            eps_lo[i,j] = lo; eps_hi[i,j] = hi
    return eps_lo, eps_hi


def main():
    print("Floquet quasi-energy near gap point (pi, 0)")
    print(f"lambda = m_pred (small tau) = {LAMBDA:.6f}")
    print()

    # Gap structure at exactly (pi, 0)
    print("=== Quasi-energy at (pi, 0) for various tau ===")
    print(f"{'tau':>8} {'eps_lo':>12} {'eps_hi':>12} {'gap/2':>12}  comment")
    for tau in [0.05, 0.1, 0.2, 0.5, 1.0]:
        lo, hi = floquet_quasienergy(PI, 0.0, tau, tau)
        gap = (hi - lo) / 2
        comment = f"BCH limit: m={LAMBDA:.4f}" if tau < 0.1 else ""
        print(f"{tau:8.3f} {lo:12.6f} {hi:12.6f} {gap:12.6f}  {comment}")
    print()

    # Numerical group velocity from quasi-energy at small offset
    print("=== Numerical group velocity vs delta (tau=1.0) ===")
    print("(uses finite-difference d(eps)/dkx around k0=(pi-delta, 0))")
    print(f"{'delta':>8} {'eps_lo':>12} {'v_floq':>10} {'v_meas (from #1)':>18}")
    for delta, vmeas in [(0.05, 0.0928), (0.10, 0.1661), (0.20, 0.2734),
                          (0.40, 0.3844), (0.80, 0.4346)]:
        h = 1e-4
        kx0 = PI - delta
        lo_p, _ = floquet_quasienergy(kx0 + h, 0.0)
        lo_m, _ = floquet_quasienergy(kx0 - h, 0.0)
        v_floq = (lo_p - lo_m) / (2 * h)  # gradient at lower band
        # account for sign: lower band, d(quasi-eps)/dk_x is the group velocity
        print(f"{delta:8.2f} {(lo_p+lo_m)/2:12.6f} {v_floq:10.4f} {vmeas:18.4f}")
    print()

    # Zoom: dispersion shape at (pi, 0)
    print("=== Quasi-energy along k_x at k_y=0 (Floquet, tau=1) ===")
    print(f"{'k_x':>10} {'eps_lower':>14} {'sqrt(m^2 + (k-pi)^2)/2':>22}")
    for kx in [PI-0.4, PI-0.2, PI-0.1, PI-0.05, PI, PI+0.05, PI+0.1, PI+0.2, PI+0.4]:
        lo, _ = floquet_quasienergy(kx, 0.0)
        # naive Lorentz prediction with effective mass m_eff and small Trotter:
        # at small tau: lo ≈ -|d| where d = (sin kx, 0, cos kx + 1 + lam cos kx)
        # but at tau=1: m_eff = ?
        m_floq, _ = floquet_quasienergy(PI, 0.0)
        # the "Lorentz fit" with measured m_floq:
        lorentz_pred = -np.sqrt(m_floq**2 + (kx - PI)**2 / 4)  # /2 for Trotter
        print(f"{kx:10.4f} {lo:14.6f} {lorentz_pred:22.6f}")


if __name__ == "__main__":
    main()
