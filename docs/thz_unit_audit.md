# THz-unit audit

## Scope

This audit compares `get_thermal_conductivity_THz_unit` with the canonical
cm^-1 implementation using the bundled `aSi512_test` input. The two functions
must represent the same Allen-Feldman calculation and differ only in their
frequency coordinate.

## Root causes

The previous THz route produced a different thermal conductivity for four
independent reasons:

1. The harmonic heat-capacity argument contained an extra factor of `1/2`.
   `omega` is already an angular frequency, so the dimensionless argument is
   `hbar * omega / (k_B * T)`.
2. `omega_threshould` and `broadening_threshould` were used without changing
   units. The first has frequency units, while the Lorentzian cutoff has
   inverse-frequency units.
3. The diffusivity prefactor did not include the complete conversion associated
   with the angular-THz frequency scale.
4. The THz route ignored `symmetrize_fc`, so it could use a different dynamical
   matrix from the cm^-1 route.

The harmonic heat-capacity evaluation also returned `0/0` at exactly zero
frequency. It now uses the analytic limit `C_i -> k_B`.

## Unit contract

The YAML parameters retain the units of the canonical cm^-1 interface:

- `omega_threshould`: cm^-1
- `broadening_factor`: dimensionless when `using_mean_spacing: True`;
  otherwise cm^-1
- `broadening_threshould`: (cm^-1)^-1

For the THz calculation, define

```text
r = (scale_THz * 2*pi) / scale_cm
```

Then the internal angular-THz values are

```text
omega_threshold_THz = omega_threshold_cm * r
fixed_broadening_THz = fixed_broadening_cm * r
lorentzian_threshold_THz = lorentzian_threshold_cm / r
```

The Allen-Feldman prefactor for an internal frequency scale `s` is

```text
pi/48 * sqrt(1e-17 * eV_J * N_A) * s^3
```

with `s = scale_cm` for the canonical route and
`s = scale_THz * 2*pi` for the THz route.

## aSi512 audit result

At 300 K with `aSi512_test/setup.yaml` and symmetrized force constants:

| calculation | summed thermal conductivity (W/m K) |
| --- | ---: |
| cm^-1 reference route before explicit mode gates | 0.9768075632 |
| corrected THz route before explicit mode gates | 0.9768075632 |
| previous THz route | 1.2759139123 |

The corrected THz and cm^-1 values agree to floating-point precision. The
previous discrepancy is reproduced by applying the fixes sequentially:

| state | summed thermal conductivity (W/m K) |
| --- | ---: |
| previous THz implementation | 1.2759139123 |
| heat-capacity correction | 1.1527073248 |
| threshold-unit correction | 1.0492816199 |
| prefactor correction | 0.9768075632 |

For the bundled configuration, the unconverted Lorentzian cutoff admitted
202,717 mode pairs, compared with 86,724 pairs under the cm^-1-equivalent
cutoff. This was the main discrete-selection difference.

## Rigid-translation and low-frequency gates

The cm^-1, THz, and direction-resolved routes use the same mode gates. For each
mass-weighted eigenvector `e_n`, pyAF evaluates its overlap with the three
mass-weighted rigid translations:

```text
p_trans[n] = sum_alpha |e_n.T t_alpha|^2
```

Gate 1 checks for negative non-translation modes. Frequencies below
`-negative_mode_tolerance_cm` cause the calculation to stop; smaller negative
non-translation frequencies are reported as `REVIEW`.

Gate 2 returns `PASS` when the three largest translation overlaps belong to the
three lowest modes, the overlap thresholds pass, and the acoustic-sum-rule
residual is sufficiently small. If this test needs review, the calculation may
continue as `PASS_WITH_CUTOFF` only when:

```text
sum(p_trans[frequency <= omega_threshould]) >= translation_capture_min
sum(p_trans[frequency > omega_threshould]) <= translation_leakage_max
```

`PASS_WITH_CUTOFF` is a cutoff-conditioned result. It can exclude genuine
physical low-frequency modes together with numerical rigid translations, so it
must not be presented as an unqualified all-mode thermal conductivity.

With the explicit mode gates enabled, the bundled symmetrized aSi512 case
returns:

| diagnostic | result |
| --- | ---: |
| Gate 1 | PASS |
| Gate 2 | PASS |
| translation capture | 3.000000000000 |
| active translation leakage | 4.97e-27 |
| cutoff-excluded modes | 3 |
| cutoff-excluded non-translation modes | 0 |
| cm^-1 thermal conductivity | 0.9800921205 W/m K |
| THz thermal conductivity | 0.9800921205 W/m K |

The value changes by about 0.34% from the pre-gate result because the three
rigid translations are now also excluded when evaluating the mean positive-mode
spacing used for Lorentzian broadening.
