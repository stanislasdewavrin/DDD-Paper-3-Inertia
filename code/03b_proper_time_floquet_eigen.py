"""
Paper III — experiment 03b: kinematic clock-rate, with proper Floquet eigenstate
================================================================================

The wavepacket spinor must be the Floquet eigenstate of U_F = U_B U_A
at the central k0, not the static H_A + H_B eigenstate, otherwise the
phase has a beat from upper/lower band mixing.
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)

PI = np.pi
LAMBDA = 1.0 / (2 * PI)


def step_A(psi, tau_A, L):
    psi_a = psi[..., 0]; psi_b = psi[..., 1]
    a_k = np.fft.fft2(psi_a); b_k = np.fft.fft2(psi_b)
    kx = 2 * PI * np.fft.fftfreq(L); ky = 2 * PI * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dx = np.sin(KX); dy = np.sin(KY); dz = np.cos(KX) + np.cos(KY)
    d_norm = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-15)
    c = np.cos(d_norm * tau_A); s = np.sin(d_norm * tau_A)
    U00 = c - 1j * s * dz / d_norm
    U01 = -1j * s * (dx - 1j * dy) / d_norm
    U10 = -1j * s * (dx + 1j * dy) / d_norm
    U11 = c + 1j * s * dz / d_norm
    new_a_k = U00 * a_k + U01 * b_k
    new_b_k = U10 * a_k + U11 * b_k
    return np.stack([np.fft.ifft2(new_a_k), np.fft.ifft2(new_b_k)], axis=-1)


def step_B(psi, tau_B, L, lam=LAMBDA):
    psi_a = psi[..., 0]; psi_b = psi[..., 1]
    a_k = np.fft.fft2(psi_a); b_k = np.fft.fft2(psi_b)
    kx = 2 * PI * np.fft.fftfreq(L); ky = 2 * PI * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    dz = lam * np.cos(KX + KY)
    phase = np.exp(-1j * dz * tau_B)
    return np.stack([np.fft.ifft2(phase * a_k),
                     np.fft.ifft2(np.conj(phase) * b_k)], axis=-1)


def floquet_diagonalize(kx, ky, tau_A=1.0, tau_B=1.0, lam=LAMBDA):
    """Return (eps_lo, eps_hi, u_lo, u_hi): quasi-energies and eigenstates."""
    dxA = np.sin(kx); dyA = np.sin(ky); dzA = np.cos(kx) + np.cos(ky)
    nA = np.sqrt(dxA**2 + dyA**2 + dzA**2 + 1e-15)
    cA = np.cos(nA * tau_A); sA = np.sin(nA * tau_A)
    UA = np.array([[cA - 1j*sA*dzA/nA, -1j*sA*(dxA - 1j*dyA)/nA],
                   [-1j*sA*(dxA + 1j*dyA)/nA, cA + 1j*sA*dzA/nA]], dtype=complex)
    dzB = lam * np.cos(kx + ky)
    UB = np.array([[np.exp(-1j*dzB*tau_B), 0],
                   [0, np.exp(1j*dzB*tau_B)]], dtype=complex)
    UF = UB @ UA
    evals, evecs = np.linalg.eig(UF)
    T = tau_A + tau_B
    quasi = -np.angle(evals) / T
    if quasi[0] < quasi[1]:
        return float(quasi[0]), float(quasi[1]), evecs[:, 0], evecs[:, 1]
    else:
        return float(quasi[1]), float(quasi[0]), evecs[:, 1], evecs[:, 0]


def make_wavepacket_floquet(L, k0, sigma, x0=None):
    """Wavepacket with Floquet lower-band eigenstate as spinor."""
    if x0 is None:
        x0 = (L / 2, L / 2)
    eps_lo, eps_hi, u_lo, u_hi = floquet_diagonalize(k0[0], k0[1])
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    env = np.exp(-((xs - x0[0])**2 + (ys - x0[1])**2) / (2 * sigma**2))
    plane = np.exp(1j * (k0[0] * xs + k0[1] * ys))
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane * u_lo[0]
    psi[..., 1] = env * plane * u_lo[1]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi, eps_lo


def centroid(psi, L):
    rho = np.abs(psi[..., 0])**2 + np.abs(psi[..., 1])**2
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    angle_x = np.angle(np.sum(rho * np.exp(2j * PI * xs / L)))
    angle_y = np.angle(np.sum(rho * np.exp(2j * PI * ys / L)))
    cx = (angle_x / (2 * PI)) * L % L
    cy = (angle_y / (2 * PI)) * L % L
    return float(cx), float(cy)


def interp_psi(psi, cx, cy, L):
    ix0 = int(np.floor(cx)) % L; ix1 = (ix0 + 1) % L
    iy0 = int(np.floor(cy)) % L; iy1 = (iy0 + 1) % L
    fx = cx - np.floor(cx); fy = cy - np.floor(cy)
    return ((1-fx)*(1-fy) * psi[ix0, iy0] +
            fx*(1-fy) * psi[ix1, iy0] +
            (1-fx)*fy * psi[ix0, iy1] +
            fx*fy * psi[ix1, iy1])


def main():
    L = 64
    sigma = 6.0
    tau_A = 1.0; tau_B = 1.0
    T_per_cycle = tau_A + tau_B
    n_cycles = 40

    eps_lo_gap, _, _, _ = floquet_diagonalize(PI, 0.0)
    m_floq = abs(eps_lo_gap)
    print(f"Floquet rest mass m = {m_floq:.6f}")
    print(f"Rest period T_rest = 2 pi / m = {2*PI/m_floq:.2f}")
    print()

    deltas = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    print(f"{'delta':>8} {'E':>10} {'gamma':>8} "
          f"{'m/gamma':>12} {'omega_meas':>12} {'rel.err':>10}  {'omega_E':>10}")
    print("(omega_E shown as cross-check: should match E itself, since")
    print(" pure lower-band -> phase rate = -eps_lo = E in lab.)")
    print()

    results = []
    for delta in deltas:
        k0 = (PI - delta, 0.0)
        eps_lo, eps_hi, u_lo, u_hi = floquet_diagonalize(k0[0], k0[1])
        E_floq = abs(eps_lo)
        gamma = E_floq / m_floq
        omega_pred_proper = m_floq / gamma  # = m^2/E

        psi, _ = make_wavepacket_floquet(L, k0, sigma)
        slow_phase = []
        # Also track phase at FIXED point (lab phase, no comoving)
        lab_phase = []
        for it in range(n_cycles + 1):
            cx, cy = centroid(psi, L)
            spinor = interp_psi(psi, cx, cy, L)
            psi_a = spinor[0]
            psi_slow = psi_a * np.exp(-1j * PI * cx)
            slow_phase.append(np.angle(psi_slow))
            # Lab phase: at FIXED center (L/2, L/2) — to verify lab freq = E
            spinor_lab = psi[L // 2, L // 2]
            lab_phase.append(np.angle(spinor_lab[0] * np.exp(-1j * PI * (L // 2))))
            if it == n_cycles:
                break
            psi = step_A(psi, tau_A, L)
            psi = step_B(psi, tau_B, L)

        slow_phase = np.unwrap(np.array(slow_phase))
        lab_phase = np.unwrap(np.array(lab_phase))
        ts = np.arange(n_cycles + 1) * T_per_cycle
        i0, i1 = 5, n_cycles - 4
        coef_slow = np.polyfit(ts[i0:i1+1], slow_phase[i0:i1+1], 1)
        coef_lab = np.polyfit(ts[i0:i1+1], lab_phase[i0:i1+1], 1)
        omega_slow = abs(float(coef_slow[0]))
        omega_lab = abs(float(coef_lab[0]))
        rel_err = (omega_slow - omega_pred_proper) / omega_pred_proper if omega_pred_proper > 0 else 0.0

        print(f"{delta:8.3f} {E_floq:10.6f} {gamma:8.4f} "
              f"{omega_pred_proper:12.6f} {omega_slow:12.6f} {rel_err:9.2%}  {omega_lab:10.6f}")

        results.append({
            "delta": float(delta), "k0x": float(k0[0]),
            "E": float(E_floq), "gamma": float(gamma),
            "omega_pred_proper": float(omega_pred_proper),
            "omega_meas_slow": float(omega_slow),
            "omega_meas_lab": float(omega_lab),
            "rel_err": float(rel_err),
        })

    out = {
        "L": L, "sigma": sigma, "n_cycles": n_cycles,
        "tau_A": tau_A, "tau_B": tau_B, "m_floq": m_floq,
        "lambda": LAMBDA, "results": results,
    }
    with open(DATA / "03b_proper_time_floquet.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved to {DATA / '03b_proper_time_floquet.json'}")


if __name__ == "__main__":
    main()
