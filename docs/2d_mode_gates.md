# 2D mode gates

## Scope

`two_dim: True` keeps the same three mass-weighted rigid translations as the
3D route. Periodicity in two directions does not remove the uniform
out-of-plane translation. The 2D route adds an out-of-plane-polarization
diagnostic for non-translation modes; it does not replace the translation
overlap test.

For normalized mass-weighted eigenvector `e_n`, pyAF evaluates

```text
p_z[n] = sum_i |e_n(i,z)|^2
```

A non-translation mode is classified as ZA-like when

```text
p_z[n] >= flexural_polarization_min
```

The default threshold is `0.8`.

## Common Gate 1 before ZA classification

Gate 1 first evaluates every negative non-translation mode, exactly as in 3D.
`negative_mode_tolerance_cm` divides them into:

| class | condition | common action |
| --- | --- | --- |
| small negative | `abs(f) <= negative_mode_tolerance_cm` | warn and review |
| significant negative | `abs(f) > negative_mode_tolerance_cm` | fail |

The code always reports the total negative-mode set, including its indices,
frequencies, out-of-plane polarizations, and the significant/small split.
The frequency cutoff is not used to turn a Gate 1 failure into a pass.

## Additional 2D polarization classification

ZA classification refines the diagnosis after the common stability check; it
does not limit Gate 1 to ZA modes. The final Gate 1 status is:

| negative-mode composition | Gate 1 status | action |
| --- | --- | --- |
| no negative non-translation mode | `PASS` | continue |
| all small negative modes are ZA-like | `REVIEW_2D_FLEXURAL` | continue with warnings |
| at least one small negative mode is non-ZA, with no significant mode | `REVIEW` | continue with warnings |
| all significant negative modes are ZA-like | `FAIL_2D_FLEXURAL` | stop |
| at least one significant negative mode is non-ZA | `FAIL` | stop |

When ZA-like and non-ZA negative modes coexist, both subsets generate
independent warnings and diagnostics. Relevant `mode_gate` fields include:

```text
significant_negative_nontranslation_count
significant_negative_nontranslation_indices
small_negative_nontranslation_count
small_negative_nontranslation_indices
negative_za_like_count
negative_za_like_indices
negative_non_za_count
negative_non_za_indices
```

## Gate 2 and the frequency cutoff

Gate 2 still identifies exactly three rigid translations. When those
translations are not uniquely the three lowest modes, but
`omega_threshould` removes their complete overlap with negligible leakage, the
2D status is `PASS_WITH_CUTOFF_2D`.

`mode_gate` separately records positive physical non-translation modes below
the cutoff. In particular:

```text
cutoff_excluded_za_mode_count
cutoff_excluded_za_mode_indices
cutoff_excluded_za_frequencies_cm
```

These fields exclude negative modes from the ZA cutoff count, so the count
answers how many positive physical ZA-like modes were discarded only because
of `omega_threshould`.

## Exploratory failure override

The default is:

```yaml
allow_2d_flexural_fail: False
```

With this default, `FAIL_2D_FLEXURAL` stops before transport is evaluated. For
an explicitly marked diagnostic calculation only, setting

```yaml
allow_2d_flexural_fail: True
```

continues with overall status `TEST_ONLY_FAIL_2D_FLEXURAL`. This is permitted
only when every significant negative mode is ZA-like. If any significant
non-ZA mode is present, Gate 1 is `FAIL` and this override has no effect.
Negative modes remain excluded from the Allen-Feldman sum. This override does
not convert an unstable structure into a physically validated one.

## Direction-resolved conductivity

`resolved_thermal_conductivity` returns a
`thermal_conductivity_summary` dictionary for 2D inputs:

```text
x
y
z_diagnostic
in_plane_average = (x + y) / 2
unit = W/mK
```

The `z_diagnostic` component is retained as a diagnostic. The reported 2D
in-plane scalar is the arithmetic mean of the two periodic directions.

## Bundled aGr exploratory result

The bundled `aGr_test` input has one significant negative ZA-like
non-translation mode. Its test configuration therefore enables the explicit
override and produces:

| diagnostic | result |
| --- | ---: |
| Gate 1 | `FAIL_2D_FLEXURAL` |
| Gate 2 | `PASS_WITH_CUTOFF_2D` |
| overall status | `TEST_ONLY_FAIL_2D_FLEXURAL` |
| negative ZA-like frequency | -12.242284 cm^-1 |
| negative-mode out-of-plane polarization | 0.977533 |
| positive ZA modes excluded only by cutoff | 1 |
| excluded positive ZA frequency | 1.131217 cm^-1 |
| kappa_x | 4.283950 W/mK |
| kappa_y | 4.214038 W/mK |
| kappa_in_plane | 4.248994 W/mK |
| kappa_z diagnostic | 0.001944 W/mK |

These values are cutoff-conditioned, test-only diagnostics. They are not an
unqualified conductivity prediction for a mechanically stable aGr structure.
