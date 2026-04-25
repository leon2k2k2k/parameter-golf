# 035h — learnable `alpha/beta` on the sparse-gate family

`035f` answered the learnable-alpha/beta question on the non-sparse family.
`035e` is currently the best branch. So the right next test is simple:

- take `035e`
- keep sparse gate
- keep the full winning training-side stack
- let recurrent `alpha/beta` learn during training

This is cleaner than mutating the meaning of `035f` after the fact.
