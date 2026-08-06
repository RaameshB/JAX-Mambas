# JAX-Mambas
This is a repo where I (attempt to) implement all of the 3 major Mamba variants in JAX + Flax's NNX API. 
Likely order of implementation:
1. Pure mathematical formulations as described in the papers
2. Stability tricks used in the official repos (may or may not get all of them)
3. Pallas implementations of their CUDA kernels

## Implementation Progress:
- [x] [Mamba](https://arxiv.org/abs/2312.00752):
  - [x] Mathematical Form
  - [x] Stability Tricks
  - [x] ~~Pallas~~ _CUDA_ Kernel (Pallas doesn't really offer a good blocked prefix scan and attempts to write a custom one have not gone well)
  - [x] LayerNorm/RMSNorm and added variable length sequence padding support
  - [x] Induction Heads replication
- [ ] [Mamba-2](https://arxiv.org/abs/2405.21060):
  - [ ] Mathematical Form
  - [ ] Stability Tricks
  - [ ] Pallas Kernel
- [ ] [Mamba-3](https://arxiv.org/abs/2603.15569):
  - [ ] Mathematical Form
  - [ ] Stability Tricks
  - [ ] Pallas Kernel

  
## Tasks on hold
- All LM tasks - I don't have the money to train a language model unfortunately
- Mamba-1 selective copying - would take over a day of training with my current setup, I cannot afford this either.
  - There is a semi-complete version of my selective copying code somewhere buried in the dev commits, but it's guaranteed to error but I'm pretty sure the config will work if the bugs are fixed. Been removed from the repo because it's not complete, might add it back in via its own branch later.


## License

The original JAX-Mambas code is available under the MIT License. The adapted
CUDA selective-scan implementation is available under Apache-2.0; see
`LICENSE-APACHE` and `NOTICE` for its upstream attribution.
