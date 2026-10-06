# ADR 0004: Reliability failures fail closed

## Status
Accepted.

## Decision
TopoCal must never accept a neural prediction when support or calibrated-risk infrastructure is unavailable, invalid, non-finite, or outside its declared numeric range.

- invalid/unavailable support -> numerical fallback
- outside support -> numerical fallback and `risk=None`
- invalid/unavailable risk -> numerical fallback and `risk=None`
- unsupported queries are **not** assigned a fictitious risk of `1.0`
- invalid fallback-solver outputs raise a hard runtime error

`RoutingResult` therefore carries explicit `RiskStatus` and `SupportStatus` values.

## Rationale
The system exists to mediate trust. Failing open on `NaN`, exceptions, schema failures, or unsupported regimes would invert its safety purpose. Outside calibration support, a numeric probability has no justified interpretation and must be represented as unknown.
