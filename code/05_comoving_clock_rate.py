"""
Paper III — experiment 05: operational comoving clock-rate
============================================================

Direct measurement of the proper-time clock rate at moving packets.

Method:
  1. Initialize a Floquet lower-band wavepacket centred at k0 = (pi-delta, 0).
  2. Track <k0|psi(t)> = sum_x exp(-i k0.x) psi_a(x, t) every cycle.
  3. The lab-frame phase rate of <k0|psi> is +E (since psi evolves as
     exp(+i E t) for the lower band).
  4. To get the COMOVING (proper-time) rate, multiply by exp(-i delta v_g t)
     where v_g = d|eps_lo|/d|p| is the smooth-envelope group velocity.
     This subtracts the Galilean shift (k0-pi) v_g = -delta v_g.
  5. The dealiased comoving phase rate = E - delta*v_g  (smooth envelope)
                                       = m^2 / E   (Lorentz, exact)
                                       = m / gamma.

For Lorentz dispersion E^2 = m^2 + (p/2)^2: v_g = p/(4E), so delta*v_g
= delta^2 / (4E), and E - delta*v_g = (E^2 - delta^2/4)/E = m^2/E.

This is the operational analogue of the bandwidth identification of
Paper II: m/gamma is the proper-time rate at the moving packet,
measured operationally (no dispersion fit required).
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
    psi_a = psi[..., 0]; psi_b = psi[..., 1]
    a_k = np.fft.fft2(psi_a); b_k = np.fft.fft2(psi_b)
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
    UB = np.array([[np.exp(-1j*dzB*tau_B), 0],[0, np.exp(1j*dzB*tau_B)]], dtype=complex)
    UF = UB @ UA
    evals, evecs = np.linalg.eig(UF)
    T = tau_A + tau_B
    quasi = -np.angle(evals)/T
    if quasi[0] < quasi[1]:
        return float(quasi[0]), float(quasi[1]), evecs[:, 0]
    return float(quasi[1]), float(quasi[0]), evecs[:, 1]


def floquet_vg(kx0, ky0, h=1e-4):
    """Numerical group velocity dε_lo/dk_x at (kx0, ky0)."""
    lo_p, _, _ = floquet_diag(kx0+h, ky0)
    lo_m, _, _ = floquet_diag(kx0-h, ky0)
    return (lo_p - lo_m) / (2*h)


def make_wavepacket_floquet(L, k0, sigma):
    eps_lo, _, u_lo = floquet_diag(k0[0], k0[1])
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    x0 = (L/2, L/2)
    env = np.exp(-((xs-x0[0])**2 + (ys-x0[1])**2) / (2*sigma**2))
    plane = np.exp(1j*(k0[0]*xs + k0[1]*ys))
    psi = np.zeros((L, L, 2), dtype=complex)
    psi[..., 0] = env * plane * u_lo[0]
    psi[..., 1] = env * plane * u_lo[1]
    psi /= np.sqrt(np.sum(np.abs(psi)**2))
    return psi, eps_lo


def project_k0(psi, k0, L):
    xs, ys = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    phase = np.exp(-1j*(k0[0]*xs + k0[1]*ys))
    A_a = np.sum(phase * psi[..., 0])
    A_b = np.sum(phase * psi[..., 1])
    return complex(A_a), complex(A_b)


def main():
    L = 128
    sigma = 12.0
    tau_A = 1.0; tau_B = 1.0
    T_per_cycle = tau_A + tau_B
    n_cycles = 60

    eps_lo_gap, _, _ = floquet_diag(PI, 0.0)
    m_floq = abs(eps_lo_gap)
    print(f"Floquet rest mass m = {m_floq:.6f}")
    print()

    # Use a finer sweep to test Lorentz form across a broad range
    deltas = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                       0.60, 0.80, 1.00, 1.20, 1.40])

    print(f"{'delta':>6} {'E_floq':>9} {'gamma':>8} {'v_g_floq':>10} "
          f"{'m/gamma_pred':>13} {'omega_co_meas':>14} {'rel.err':>9} {'reg':>6}")
    print()

    results = []
    for delta in deltas:
        k0 = (PI - delta, 0.0)
        eps_lo, _, _ = floquet_diag(k0[0], k0[1])
        E_floq = abs(eps_lo)
        gamma_floq = E_floq / m_floq
        v_g_floq = floquet_vg(k0[0], 0.0)  # smooth-envelope vg
        m_over_gamma = m_floq / gamma_floq

        psi, _ = make_wavepacket_floquet(L, k0, sigma)
        # Track <k0|psi(t)> and apply Galilean dealiasing
        proj_lab_phase = []
        for it in range(n_cycles + 1):
            A_a, A_b = project_k0(psi, k0, L)
            A = A_a if abs(A_a) > abs(A_b) else A_b
            proj_lab_phase.append(A)
            if it == n_cycles:
                break
            psi = step_A(psi, tau_A, L)
            psi = step_B(psi, tau_B, L)

        ts = np.arange(n_cycles + 1) * T_per_cycle
        # Dealias: multiply each <k0|psi(t)> by exp(-i delta v_g t)
        # This shifts to comoving smooth-envelope frame.
        proj_co = np.array(proj_lab_phase) * np.exp(-1j * delta * v_g_floq * ts)

        # Phase tracking with unwrap
        ph = np.unwrap(np.angle(proj_co))
        i0, i1 = 5, n_cycles - 4
        slope = np.polyfit(ts[i0:i1+1], ph[i0:i1+1], 1)[0]
        omega_co_meas = abs(float(slope))

        # Lorentz regime indicator: |delta| < |2m| means within Lorentz limit
        regime = "Lor" if delta < 2*m_floq*4 else "lat"  # rough threshold

        rel_err = (omega_co_meas - m_over_gamma) / m_over_gamma if m_over_gamma > 0 else 0.0

        print(f"{delta:6.3f} {E_floq:9.6f} {gamma_floq:8.4f} {v_g_floq:10.6f} "
              f"{m_over_gamma:13.6f} {omega_co_meas:14.6f} {rel_err:8.2%} {regime:>6}")

        results.append({
            "delta": float(delta),
            "k0x": float(k0[0]),
            "E_floq": float(E_floq),
            "gamma_floq": float(gamma_floq),
            "v_g_floq": float(v_g_floq),
            "m_over_gamma_pred": float(m_over_gamma),
            "omega_comoving_meas": omega_co_meas,
            "rel_err": float(rel_err),
        })

    out = {"L": L, "sigma": sigma, "n_cycles": n_cycles,
           "tau_A": tau_A, "tau_B": tau_B, "m_floq": m_floq,
           "lambda": LAMBDA, "results": results}
    with open(DATA / "05_comoving_clock.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Saved to {DATA / '05_comoving_clock.json'}")


if __name__ == "__main__":
    main()
