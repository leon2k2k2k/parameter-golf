# Idea 035f — learnable `alpha/beta` on the `#1779` / `030` family

## Core idea

Take the strongest frozen `alpha/beta` family we have and unfreeze the carry
coefficients during normal training, while keeping their strong `025b`
initialization.

So:

- same family as `030`
- same `alpha/beta` structure
- same initialization
- but trainable instead of frozen

## Why this is interesting

Right now we are mostly testing recipe changes on top of a frozen carry.

This spec asks a simpler structural question:

- are the old frozen coefficients still the right ones for the modern stack?

If not, allowing them to move may reveal:

- a better calibrated `alpha/beta` setting
- or that the frozen values are already near-optimal

## First question

On a `4×H` screen:

- does learnable `alpha/beta` beat the old `026` reference `1.06770372`?

## Important design choice

Do **not** randomize or neutralize the carry.

Use the existing `025b` values as the start point, then let training decide how
far to move from there.
