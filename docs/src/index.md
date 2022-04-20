```@meta
CurrentModule = Dedisp
```

# Dedisp

Documentation for [Dedisp](https://github.com/kiranshila/Dedisp.jl).

The main exported method is `dedisp` which performs the incoherent dedispersion. TODO write more here.

For the GPU-side, there is a frequency-chunked version which sums the various frequency channels in parallel then does a block reduction. For certain sizes of input, this may be more performant, but the frequency axis has to be quite big to get over the block scheduling overhead.

```@contents
Pages = ["api.md"]
Depth = 3
```
