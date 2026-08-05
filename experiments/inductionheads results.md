# Induction Heads Results
After constructing a paper-faithful model, which involves the following deviations from existing implementations…
- using fixed broadcasting and not the generalized “learned” broadcasting via a low-rank projection that was floated in the paper and also implemented as the default (I believe?) in the official repo
- Full ZOH discretization instead of using the Exponential Euler approximation used in the official repo
- No GPT-Style Residual scaling
- LayerNorm instead of RMSNorm
- No feedthrough learned (not selected) diagonal D matrix present in the S6 block 
… we get some very nice results, and paper-accurate results:
	```
	Seq Len 256 Accuracy: 100.00
	Seq Len 512 Accuracy: 100.00
	Seq Len 1024 Accuracy: 100.00
	Seq Len 2048 Accuracy: 100.00
	Seq Len 4096 Accuracy: 100.00
	Seq Len 8192 Accuracy: 100.00
	Seq Len 16384 Accuracy: 100.00
	Seq Len 32768 Accuracy: 100.00
	```
This was trained on a fixed sequence length of 256, so it never saw these longer sequence lengths during training. Evaluations are averaged over 128 samples.

I did not train run eval on sequences longer than ~32k due to running out of memory on my A100 from Colab, I’m sure that in the future, longer sequences could be fit using the recurrent formulation of Mamba, but I haven’t implemented a recurrent batch generation algorithm yet. Even better, chunked parallelism would probably be the most efficient for this, but I don’t quite know how to do that for the Mamba block as a whole.

I have a journal regarding the other experiments I’ve done, along with my many failings in replicating this experiment (the configuration used to replicate it is in the experiment’s train script). This will be added in future commits.