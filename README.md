# Multi-Pass Transformer Training
This project explores a way to train transformers for recurrent-style inference without training them as recurrent models over token time. The models are trained with multiple passes over the same token-sequence. Earlier passes write per-token memory states; later passes read shifted versions of those memories, giving each token access to deep-layer information from previous token positions while preserving parallel training.

## Early Testing
Transformers have a notoriously limited ability to track state (see for example [Li25](https://arxiv.org/abs/2503.02854)), but with multi-pass training and attention over a memory-tape at inference-time we get much improved performance. The permutation task looks like "STATE [A, B, C, D], swap 1 2, STATE [B, A, C, D]" with one swap. I trained the models with a curriculum of increasing difficulty; when the model learns to predict STATE after $n$ swaps with an accuracy of 95% (or 100\%) it "graduates" to $n+1$ swaps.  

![](figures/permutation_task_plot.png "Permutation Task")

For this experiment both baseline transformers are tiny 4 layer, 4 attention-head and 128 embedding dimension models. The baseline transformer is severely depth-constrained whereas additional attention to memory tokens enables the multi-pass model to recurrently track the state.


## Motivation
The training-time parallelization of decoder-only transformers is one of the main reasons they scale so well. At layer $l$, the hidden state $h_i^l$ at position $i$ can attend to earlier positions $h_{j<i}^{l-1}$ from layer $l-1$, and not to hidden states from the same or deeper layers. This *causal* attention pattern permits hidden states for all token positions in a layer $[h_{1}^{l}, \ldots, h_{n}^l]$ to be computed in parallel during training, but it also disallows attention to previous tokens' deeper-layer hidden states at inference time.

This information flow makes autoregressive inference *stateless*. At each generation step the transformer predicts the next token as if seeing the context for the first time. KV caching allows the model to predict using only the current token and its cache, but the cache only improves efficiency, and it does not carry forward any information that would not be recomputed had you just fed the context to the model.

![](figures/inference_pattern_fig.png "Inference Patterns")

The tempting 'fix' would be to let the hidden state $h_{i}^{l}$ at token $i$ depend directly on the hidden state $h_{j}^{\ell}$ at token $j<i$ in the same or deeper layers $\ell \geq l$. But that would introduce a token-time recurrence: position $i$ would have to wait for position $i-1$, and the training-time parallelism would be lost.

![](figures/generation_fig.png "Generation")

But here is an idea: what if we run multiple parallel passes over the same teacher-forced sequence instead of making token $i$ wait for token $i-1$ during training? Pass 1 writes a memory tape. Pass 2 reads a shifted version of that tape. Pass 3 can read the shifted tape from pass 2, and so on. The hope is that such multi-pass training can teach the model to emit memories that are useful and stable enough to support cheaper recurrent-style memory use at inference time.


## Setup
The goal is not merely to cache attention keys and values. The goal is to train the model to read and write a memory state for each token, and then test whether those memories can be reused during generation. There are many possible memory designs. This project focuses on one memory vector per token per pass. For a token sequence
$T_{1:n} = [t_1, \ldots, t_n]$, define the pass-$i$ memory tape:

$$
M_{1:n}^i = [m_1^i, \ldots, m_n^i]
$$

The multi-pass recurrence has this shape:

```math
M_{1:n}^0 = [0, \ldots, 0]
```

```math
(\ell_{1:n}^k, M_{1:n}^k)
= F_\theta(T_{1:n}, \mathrm{shift\_right}(M_{1:n}^{k-1})),
\quad k = 1, \ldots, K.
```

Here $F_\theta$ should be read schematically: the model also produces logits, and the memory could be a learned projection of an internal hidden state rather than the hidden state itself. The important causal constraint is that token $j$ at pass $i+1$ may read memory from earlier token positions at pass $i$, but not its own previous-pass memory $m_j^i$. In batched training this is implemented by shifting the previous pass memory tape one position to the right:

```math
\mathrm{shift\_right}(M^i) := [0, m_1^i, \ldots, m_{n-1}^i]
```
That keeps the training computation parallel over token positions while giving each pass access to information written by the previous pass.


## Multi-pass Training
Multi-pass training runs the same teacher-forced sequence through the model several times. The recurrence is over the pass dimension, not over token time. That distinction is the trick; within each pass, all token positions are still computed in parallel, as in an ordinary transformer.

Perhaps the easiest way to illustrate this is to imagine using a transformed last-layer hidden state as the memory. That memory can be fed back into the next pass in different ways, for example by concatenating it to the input stream or by attending to it through a separate memory attention path.

![](figures/multipass_training_fig.png "Multi-pass Training")

The general training loop for `K` passes looks like this:

```text
M_prev = zero_memory

for k = 1..K:
    memory_in = shift_right(M_prev)
    X_k       = Decoder(tokens, memory = memory_in)
    logits_k  = LMHead(X_k)
    M_k       = MemHead(LN_mem(X_k))
    loss_k    = LM_loss(logits_k, targets)
    M_prev    = M_k
```

The total objective is a weighted sum of the per-pass language-modeling losses:

```text
TotalLoss = sum_k lambda_k * loss_k
```

Most experiments put the heaviest weight on the final pass. The final pass is therefore trained to do the main predictive work, while earlier passes are encouraged to write memories that make later predictions easier.

The training schema:

1. pass `k` reads the shifted memory tape written by pass `k - 1`
2. pass `k` predicts the same next-token targets as the other passes
3. pass `k` writes a new memory tape for pass `k + 1`

is exact with respect to this `K`-pass model. No approximation has been introduced yet.


## Mismatch and Greedy Inference

And how do we get stateful inference out of this? Well, the exact inference procedure for this model is expensive. For every new token, we can run all $K$ passes on the full current prefix. That exact recomputation preserves the same pass-by-pass recurrence used in training, but it is too expensive for the target inference mode. What we want is greedy final-pass inference:

1. Run the prompt exactly for $K$ passes.
2. Cache the prompt memory tape from pass $K-1$.
3. For each generated token, run only the final pass over the current prefix.
4. Append the memory written by that final pass to the memory cache.

![](figures/mismatch_fig.png "Training and generation mismatch")

The first generated token is special. Suppose the prompt has length $n$. When predicting $t_{n+1}$, the final pass can read the correct prompt memory tape $M_{1:n}^{K-1}$. So the first greedy step uses the same previous-pass prompt memory that exact $K$-pass inference would use. The mismatch starts after that. Once token $t_{n+1}$ has been generated, exact multi-pass inference would in theory need the pass-$K-1$ memory for that new token $m_{n+1}^{K-1}$. But greedy final-pass inference did not run pass $K-1$ for the generated token. It only ran pass $K$, so the memory it actually has is $m_{n+1}^{K}$. Therefore the exact memory tape for the next step would be:

```math
[m_1^{K-1}, \ldots, m_n^{K-1}, m_{n+1}^{K-1}]
```

while the greedy memory tape is:

```math
[m_1^{K-1}, \ldots, m_n^{K-1}, m_{n+1}^{K}]
```

That substitution is the whole approximation. After prompt warmup, greedy inference treats newly generated final-pass memories as if they were next-step previous-pass memories. The project therefore depends on a stability question:

> Does multi-pass training make adjacent-pass memories similar enough that $m_t^K$ can stand in for $m_t^{K-1}$ during generation?

If yes, generation can pay for the $K$-pass computation once on the prompt, then continue by updating only the final-pass memory approximation. If no, the greedy memory cache drifts away from the exact multi-pass model and we would have to add some auxiliary loss to force similarity. The real empirical question is the gap between exact recomputation and greedy final-pass inference.

## Multi-pass Architectures
The following architectures explore some different ways of passing on the memories between passes. They all follow the abstract multi-pass training and inference-time methods (see the parent-class `MultiPassTransformer` in the codebase). 

### Memory Through Attention: The MemoryTape Architecture 

The token stream remains an ordinary causal decoder transformer. Each layer also gets a separate causal cross-attention path into the shifted memory tape:
```text
S = SelfAttn(X, causal_token_mask)
C = MemoryCrossAttn(X, shift_right(M^(k-1)), causal_memory_mask)
Y = X + S + C
X = Y + MLP(Y)
```

The memory tape is not concatenated with the token sequence. It is a separate stream accessed only through cross-attention. The recurrent memory is also not forced to be the raw hidden state. The current implementation writes memory through a separate head:
```text
m_t^k = MemHead(LN_mem(h_t^k))
```

This lets the model learn a representation that is useful for recurrence, not just next-token prediction.

### Memory Through Embedding Concatenation: The MemoryConcat Architecture

The concat variant removes the separate memory cross-attention path. Instead, each pass builds its input stream by concatenating the token embedding with the shifted previous-pass memory at the same position, then projecting back to the model width:

```text
X_in = Fuse([TokenEmb(T), shift_right(M^(k-1))])
X    = DecoderBlocks(X_in)
```

The token stream is still present at every pass, so next-token logits are produced in the usual way. The memory is injected before the ordinary causal decoder stack rather than read by a separate attention module. This is a useful ablation for testing whether the gain comes from a recurrent memory signal itself, or specifically from content-addressed memory attention.

The implementation initializes the fusion projection so the token half starts near an identity map and the memory half starts small. This keeps the initial model close to a normal transformer while allowing training to learn how much memory to use.

### Memory as State: The MemoryUpdate Architecture

The memory-update variant makes the recurrent memory stream the main object being transformed. Instead of starting from token representations and then adding memory, each pass starts from the shifted previous-pass memory at each position, seeds it with the current token embedding, and then asks how the token history should update that memory:

```text
M_in  = LN(shift_right(M^(k-1))) + TokenToMemory(LN(TokenEmb(T)))
Delta = TokenCrossAttn(query=M_in, key/value=TokenEmb(T), causal_token_mask)
M     = M_in + Delta
M     = MemoryDecoderBlocks(M)
```

The training CLI also has an optional gated write variant via `--memory-update-gate on`:

```text
Gate = sigmoid(W_gate([M_in, TokenEmb(T), Delta]))
M    = M_in + Gate * Delta
```

This keeps gating as an ablation rather than the defining feature of the architecture. The main state-tracking view remains:

```text
new_state = update(previous_state, current_observation)
```

The model still predicts next-token logits from the updated memory stream, and writes the next recurrent memory through a separate memory head:

```text
logits_t^k = LMHead(LN(m_t^k))
M_t^k      = MemHead(LN_mem(m_t^k))
```

The implementation initializes the token-to-memory projection as an identity map, so the first pass has a useful token signal even though the incoming memory is zero. When the gate is enabled, the gate bias starts slightly negative, making early writes conservative while still learnable.


## Tasks

The current experiments focus on synthetic algorithmic tasks featuring state-tracking where exactness is easy to measure and computational "depth" is required.

### Permutation Composition

The model receives symbolic permutation operations and must predict the composed result. This tests whether the recurrent memory path helps with iterative state updates.

### REPL Traces

The model learns simple assignment-and-print traces with variable state. This is a small proxy for recurrent symbolic execution.

The available architectures are `transformer`, `memory_tape`, `memory_concat`, and `memory_update`.
Both training scripts use curriculum training. `--max-num-swaps` and `--train-program-length` set the maximum curriculum levels.
Pass `--log-jsonl path/to/log.jsonl` to write structured run and eval events.

### Architecture Examples

Baseline transformer:

```bash
python3 train_permutation.py \
  --architecture transformer \
  --curriculum-threshold 1.0 \
  --train-steps 50000
```

MemoryTape:

```bash
python3 train_permutation.py \
  --architecture memory_tape \
  --n-pass 4 \
  --pass-loss-weights 0.0 0.1 1.0 1.0 \
  --curriculum-threshold 1.0 \
  --train-steps 50000
```

MemoryConcat:

```bash
python3 train_permutation.py \
  --architecture memory_concat \
  --n-pass 4 \
  --pass-loss-weights 0.0 0.1 1.0 1.0 \
  --curriculum-threshold 1.0 \
  --train-steps 50000
```

MemoryUpdate:

```bash
python3 train_repl.py \
  --architecture memory_update \
  --n-pass 3 \
  --pass-loss-weights 0.0 0.1 1.0 \
  --memory-update-gate on \
  --randomize-num-vars \
  --train-steps 100000
```

## To Do:
1. Test scaling of the architectures: more parameters, more data, harder tasks. 
2. If there is significant divergence between $m_i^{K-1}$ and $m_i^{K}$, explore options to force similarity.
3. I should explore more systematically what the best ways to pass memory forwards is; attention? embeddings?, a small encoder?
4. Experiment with gating especially for the memory-update architecture. 

## Requirements

The code is written in Python and PyTorch. The default device is `mps`, so the current scripts are set up for Apple Silicon by default.

For local development, install the test dependency group if you want to run pytest:

```bash
python3 -m pip install ".[test]"
```

To run on CPU or CUDA, pass `--device cpu` or `--device cuda` to the training scripts.

Local reference PDFs can live under `papers/`; that directory is ignored by git.
