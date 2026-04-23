# Idea 035b — plateau on the original `030` alpha/beta family

## Thesis

`MIN_LR=0.10` is the simplest schedule transfer to test on the stronger
`030` family, but the more targeted version is still interesting:

- keep the original `030` `4×H` screen stack fixed
- add a short LR plateau starting exactly at loop onset
- see whether targeted transition support beats a global hotter tail

## Target

Primary `4×H` benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

Direct schedule comparison:

- `035` (`MIN_LR=0.10`) once available

## First question

Can a plateau with paused-time semantics on the original `030` alpha/beta
family beat `1.06770372` in `4×H` screen form, and is it competitive with
`035`?
