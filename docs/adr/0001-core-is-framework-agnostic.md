# ADR 0001: Keep TopoCal Core Framework-Agnostic

Status: Accepted

## Context

Research requires PyTorch neural operators, topology libraries, mesh tooling, and numerical solvers, but forcing all users to install them would make the reliability layer brittle and hard to test.

## Decision

Core depends only on NumPy and protocol-based interfaces. Heavy scientific libraries live in optional adapters.

## Consequences

Positive:

- lightweight tests and CI
- easier integration with different operator stacks
- architecture-independent research claims

Negative:

- adapter code is more explicit
- some shared tensor/mesh conveniences cannot leak into core types
