A Transformer-style attention layer that can:
Iteratively refine its output for each window of the sequence.
Work on long sequences by breaking them into windows.
Use custom, learnable projections for each head.
Stop early if the output isn’t changing much.
Handle both self-attention and cross-attention.

Iterative refinement might help the model “think harder” about each chunk of the sequence, potentially leading to better representations.
Sliding window lets you scale to long sequences without blowing up your GPU.
Dynamic threshold is a clever way to save compute if the answer is already “good enough.”
Custom projections per head could let each head specialize more.
