from dataclasses import asdict, dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
 

#------------------------------------------------------------
#--             Regular Causal Transformer                 --
#------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p = 0.0,
                is_causal=True
            )  # (B,H,T,hs)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B,H,T,T)
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B,H,T,hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)   # (B,T,C)
        y = self.c_proj(y)
        return y 

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd) 
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class CausalTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            ln_f=LayerNorm(config.n_embd)
        )
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def calc_loss(self, logits, targets): 
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss
    
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # Encoding...
        x = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = x + self.transformer.wpe(pos)
        # Applying blocks...
        for block in self.transformer.h:
            x = block(x)
        # Unembedding..
        logits = self.lm_head(self.transformer.ln_f(x))
        return logits
 
    def generate(
            self, 
            ids: torch.Tensor,
            max_new_tokens:int, 
            temperature=1.0, 
            do_sample=True,
            inference_mode="recompute",
            cache_source="penultimate",
        ):
        if inference_mode != "recompute":
            raise ValueError("CausalTransformer only supports inference_mode='recompute'")
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            ids_cond = ids if ids.size(1) <= self.config.block_size else ids[:, -self.config.block_size:]
            logits = self.forward(ids_cond)
            last_token_logits = logits[:, -1, :] / temperature
            if do_sample:
                probs = F.softmax(last_token_logits, dim=-1)
                ids_next = torch.multinomial(probs, num_samples=1)
            else:
                ids_next = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, ids_next), dim=1)
        return ids

@dataclass
class TransformerConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        return cls(**d)


#------------------------------------------------------------
#--          MultiPass Transformer (Parent Class)          --
#------------------------------------------------------------

class MultiPassTransformer(nn.Module):
    block_cls = None

    """Shared outer loop for recurrent multi-pass transformer variants.

    Why this fits:
    the recurrent architectures all share the same outer recurrence:

        prev_memory = zeros
        for pass_idx in range(n_pass):
            memory_tape = shift_right(prev_memory)
            hidden = run_architecture_specific_pass(token_stream, memory_tape)
            logits, memory = project_outputs(hidden)
            prev_memory = memory

    They differ in what one architecture-specific pass means. MemoryTape lets the
    token stream attend to a shifted memory tape through cross-attention.
    MemoryConcat fuses token embeddings and shifted memory at the input.
    MemoryUpdate treats shifted memory as the main stream, seeds it with token
    embeddings, then updates it through token-history attention.

    So this parent owns multi-pass scheduling and generation loops; subclasses
    own the question, "what does one pass mean?"
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.n_pass >= 2
        self.config = config

        if self.block_cls is None:
            raise ValueError(f"{type(self).__name__} must define block_cls")

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([self.block_cls(config) for _ in range(config.n_layer)]),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            ln_f=LayerNorm(config.n_embd),
        )
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ln_mem = LayerNorm(config.n_embd)
        self.mem_head = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def _finish_initialization(self):
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def calc_loss(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

    def calc_total_loss(self, logits_per_pass, targets, loss_weights=None):
        if loss_weights is None:
            loss_weights = [1.0] * len(logits_per_pass)
        if len(loss_weights) != len(logits_per_pass):
            raise ValueError("loss_weights must match number of recurrent passes")
        losses = [self.calc_loss(logits, targets) for logits in logits_per_pass]
        total_loss = sum(weight * loss for weight, loss in zip(loss_weights, losses))
        return total_loss, losses

    def _embed_tokens(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        x = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = x + self.transformer.wpe(pos)
        return x

    def forward_passes(self, idx):
        """Return logits and memory states for every recurrent pass."""
        token_stream = self._embed_tokens(idx)
        prev_memory_states = torch.zeros_like(token_stream)
        logits_per_pass = []
        memory_per_pass = []

        for _ in range(self.config.n_pass):
            memory_tape = torch.zeros_like(prev_memory_states)
            memory_tape[:, 1:, :] = prev_memory_states[:, :-1, :]
            hidden_states = self._run_full_pass(token_stream, memory_tape)
            logits = self.lm_head(self.transformer.ln_f(hidden_states))
            memory_states = self.mem_head(self.ln_mem(hidden_states))
            logits_per_pass.append(logits)
            memory_per_pass.append(memory_states)
            prev_memory_states = memory_states

        return logits_per_pass, memory_per_pass

    def forward(
            self,
            idx,
            return_all_logits=False,
        ):
        logits_per_pass, _ = self.forward_passes(idx)
        return logits_per_pass if return_all_logits else logits_per_pass[-1]

    def generate(
            self,
            ids: torch.Tensor,
            max_new_tokens: int,
            temperature=1.0,
            do_sample=True,
            inference_mode="recompute",
            cache_source="penultimate",
        ):
        if inference_mode == "recompute":
            for _ in range(max_new_tokens):
                ids_cond = ids if ids.size(1) <= self.config.block_size else ids[:, -self.config.block_size:]
                logits = self.forward(ids_cond)
                last_token_logits = logits[:, -1, :] / temperature
                if do_sample:
                    probs = F.softmax(last_token_logits, dim=-1)
                    ids_next = torch.multinomial(probs, num_samples=1)
                else:
                    ids_next = torch.argmax(last_token_logits, dim=-1, keepdim=True)
                ids = torch.cat((ids, ids_next), dim=1)
            return ids
        if inference_mode == "final_pass":
            return self._generate_final_pass(
                ids,
                max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                cache_source=cache_source,
            )
        raise ValueError(f"Unsupported inference_mode: {inference_mode}")

    def _generate_final_pass(
            self,
            ids: torch.Tensor,
            max_new_tokens: int,
            temperature=1.0,
            do_sample=True,
            cache_source="penultimate",
        ):
        if max_new_tokens <= 0:
            return ids
        if cache_source == "penultimate":
            cache_pass_idx = self.config.n_pass - 2
        elif cache_source == "last":
            cache_pass_idx = self.config.n_pass - 1
        else:
            raise ValueError("cache_source must be 'penultimate' or 'last'")

        ids_window = ids if ids.size(1) <= self.config.block_size else ids[:, -self.config.block_size:]
        token_stream = self._embed_tokens(ids_window)
        prev_memory_states = torch.zeros_like(token_stream)
        final_pass_memory_history = None

        for pass_idx in range(self.config.n_pass):
            memory_tape = torch.zeros_like(prev_memory_states)
            memory_tape[:, 1:, :] = prev_memory_states[:, :-1, :]
            hidden_states = self._run_full_pass(token_stream, memory_tape)
            logits = self.lm_head(self.transformer.ln_f(hidden_states))
            memory_states = self.mem_head(self.ln_mem(hidden_states))
            if pass_idx == cache_pass_idx:
                final_pass_memory_history = memory_states
            prev_memory_states = memory_states

        for _ in range(max_new_tokens):
            last_token_logits = logits[:, -1, :] / temperature
            if do_sample:
                probs = F.softmax(last_token_logits, dim=-1)
                ids_next = torch.multinomial(probs, num_samples=1)
            else:
                ids_next = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, ids_next), dim=1)

            ids_window = ids if ids.size(1) <= self.config.block_size else ids[:, -self.config.block_size:]
            token_stream = self._embed_tokens(ids_window)
            memory_tape = torch.zeros_like(token_stream)
            if token_stream.size(1) > 1:
                memory_tape[:, 1:, :] = final_pass_memory_history[:, -(token_stream.size(1) - 1):, :]
            hidden_states = self._run_full_pass(token_stream, memory_tape)
            logits = self.lm_head(self.transformer.ln_f(hidden_states))
            memory_states = self.mem_head(self.ln_mem(hidden_states))
            final_pass_memory_history = torch.cat((final_pass_memory_history, memory_states[:, -1:, :]), dim=1)
            final_pass_memory_history = final_pass_memory_history[:, -self.config.block_size:, :]

        return ids

@dataclass
class MultiPassConfig(TransformerConfig):
    n_pass: int

    def __post_init__(self):
        super().__post_init__()
        if self.n_pass < 2:
            raise ValueError(f"n_pass ({self.n_pass}) must be at least 2 for multi-pass models")


@dataclass
class MemoryTapeConfig(MultiPassConfig):
    memory_tape_gate: str = "tanh"

    @staticmethod
    def normalize_gate(memory_tape_gate) -> str:
        if memory_tape_gate is None:
            return "none"
        normalized = str(memory_tape_gate).lower()
        options = ("none", "tanh", "scalar")
        if normalized not in options:
            raise ValueError(f"memory_tape_gate must be one of: {', '.join(options)}")
        return normalized

    def __post_init__(self):
        super().__post_init__()
        self.memory_tape_gate = self.normalize_gate(self.memory_tape_gate)


#------------------------------------------------------------
#--               MemoryConcat Transformer                 --
#------------------------------------------------------------

class MemoryConcatTransformer(MultiPassTransformer):
    """Recurrent model that feeds shifted memory by concatenating it with token embeddings."""
    block_cls = Block

    def __init__(self, config):
        super().__init__(config)
        self.mem_in_ln = LayerNorm(config.n_embd)
        self.mem_fuse = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        self._finish_initialization()
        self._init_memory_fuse_weights()

    def _init_memory_fuse_weights(self):
        with torch.no_grad():
            n_embd = self.config.n_embd
            self.mem_fuse.weight.zero_()
            eye = torch.eye(
                n_embd,
                device=self.mem_fuse.weight.device,
                dtype=self.mem_fuse.weight.dtype,
            )
            self.mem_fuse.weight[:, :n_embd].copy_(eye)
            torch.nn.init.normal_(self.mem_fuse.weight[:, n_embd:], mean=0.0, std=0.02)

    def _fuse_token_and_memory(self, token_stream, memory_tape):
        if token_stream.shape != memory_tape.shape:
            raise ValueError("token_stream and memory_tape must have the same shape for concat fusion")
        memory_tape = self.mem_in_ln(memory_tape)
        return self.mem_fuse(torch.cat((token_stream, memory_tape), dim=-1))

    def _run_full_pass(self, token_stream, memory_tape):
        x = self._fuse_token_and_memory(token_stream, memory_tape)
        for block in self.transformer.h:
            x = block(x)
        return x


#------------------------------------------------------------
#--                 MemoryTape Transformer                 --
#------------------------------------------------------------

class CausalCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, memory):
        B, T, C = x.size()
        Bm, Tm, Cm = memory.size()
        if (B, C) != (Bm, Cm):
            raise ValueError("x and memory must share batch size and embedding dimension")
        if Tm != T:
            raise ValueError("x and memory must share sequence length in this first recurrent design")

        q = self.c_q(x)
        k, v = self.c_kv(memory).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, Tm, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, Tm, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(torch.triu(torch.ones(T, Tm, device=x.device), 1).bool(), float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    
class MemoryBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_mem_q = LayerNorm(config.n_embd)
        self.ln_mem_kv = LayerNorm(config.n_embd)
        self.cross_attn = CausalCrossAttention(config)
        self.memory_tape_gate = config.memory_tape_gate
        self.memory_gate = None if self.memory_tape_gate == "none" else nn.Parameter(
            torch.tensor(0.2 if self.memory_tape_gate == "scalar" else 0.0)
        )
        self._memory_gate_scale = {
            "none": self._ungated_memory_scale,
            "tanh": self._tanh_memory_scale,
            "scalar": self._scalar_memory_scale,
        }[self.memory_tape_gate]

    def _ungated_memory_scale(self):
        return 1.0

    def _tanh_memory_scale(self):
        return torch.tanh(self.memory_gate)

    def _scalar_memory_scale(self):
        return self.memory_gate

    def memory_gate_stats(self):
        if self.memory_gate is None:
            return None
        raw = self.memory_gate.detach()
        effective = self._memory_gate_scale().detach()
        return {
            "mode": self.memory_tape_gate,
            "raw": float(raw.cpu().item()),
            "effective": float(effective.cpu().item()),
        }

    def forward(self, x, memory_states):
        x = x + self.attn(self.ln_1(x))
        memory_delta = self.cross_attn(
            self.ln_mem_q(x),
            self.ln_mem_kv(memory_states),
        )
        x = x + self._memory_gate_scale() * memory_delta
        x = x + self.mlp(self.ln_2(x))
        return x

class MemoryTapeTransformer(MultiPassTransformer):
    block_cls = MemoryBlock

    def __init__(self, config):
        object.__setattr__(self, "_rng_state_before_construction", torch.get_rng_state())
        super().__init__(config)
        self._finish_initialization()
        del self._rng_state_before_construction

    def _causal_transformer_apply_start_rng_state(self):
        current_state = torch.get_rng_state()
        torch.set_rng_state(self._rng_state_before_construction)
        try:
            nn.ModuleDict(
                dict(
                    wte=nn.Embedding(self.config.vocab_size, self.config.n_embd),
                    h=nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)]),
                    wpe=nn.Embedding(self.config.block_size, self.config.n_embd),
                    ln_f=LayerNorm(self.config.n_embd),
                )
            )
            nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
            return torch.get_rng_state()
        finally:
            torch.set_rng_state(current_state)

    def _init_active_path_weights(self):
        self._init_weights(self.transformer.wte)
        for block in self.transformer.h:
            block.attn.apply(self._init_weights)
            block.mlp.apply(self._init_weights)
        self._init_weights(self.transformer.wpe)
        self._init_weights(self.lm_head)

    def _init_active_path_residual_projections(self):
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for block in self.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=std)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=std)

    def _init_memory_path_weights(self):
        for block in self.transformer.h:
            block.cross_attn.apply(self._init_weights)
        self._init_weights(self.mem_head)

    def _init_memory_path_residual_projections(self):
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for block in self.transformer.h:
            torch.nn.init.normal_(block.cross_attn.c_proj.weight, mean=0.0, std=std)

    def _finish_initialization(self):
        torch.set_rng_state(self._causal_transformer_apply_start_rng_state())
        self._init_active_path_weights()
        self._init_active_path_residual_projections()
        self._init_memory_path_weights()
        self._init_memory_path_residual_projections()

    def memory_gate_stats(self):
        block_stats = [
            stats
            for block in self.transformer.h
            for stats in [block.memory_gate_stats()]
            if stats is not None
        ]
        if not block_stats:
            return None

        raw_values = [stats["raw"] for stats in block_stats]
        effective_values = [stats["effective"] for stats in block_stats]
        effective = torch.tensor(effective_values)
        modes = [stats["mode"] for stats in block_stats]
        mode = modes[0] if len(set(modes)) == 1 else "mixed"
        return {
            "mode": mode,
            "raw": raw_values,
            "effective": effective_values,
            "mean_abs_effective": float(effective.abs().mean().item()),
            "max_abs_effective": float(effective.abs().max().item()),
        }

    def _run_full_pass(self, token_stream, memory_tape):
        x = token_stream
        for block in self.transformer.h:
            x = block(x, memory_tape)
        return x

#------------------------------------------------------------
#--               MemoryUpdate Transformer                 --
#------------------------------------------------------------

class MemoryUpdateBlock(nn.Module):
    """Memory-stream block where memory queries token evidence and optionally gates the write."""

    def __init__(self, config):
        super().__init__()
        self.ln_mem_q = LayerNorm(config.n_embd)
        self.ln_tok_kv = LayerNorm(config.n_embd)
        self.token_attn = CausalCrossAttention(config)
        self.use_memory_gate = config.use_memory_gate
        self.token_gate = nn.Linear(3 * config.n_embd, config.n_embd, bias=True) if self.use_memory_gate else None
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def set_gate_bias(self, bias: float):
        if self.token_gate is not None:
            nn.init.constant_(self.token_gate.bias, bias)

    def _apply_token_update(self, memory_states, token_stream, token_delta):
        if not self.use_memory_gate:
            return memory_states + token_delta
        gate_input = torch.cat((memory_states, token_stream, token_delta), dim=-1)
        gate = torch.sigmoid(self.token_gate(gate_input))
        return memory_states + gate * token_delta

    def forward(self, memory_states, token_stream):
        token_delta = self.token_attn(
            self.ln_mem_q(memory_states),
            self.ln_tok_kv(token_stream),
        )
        memory_states = self._apply_token_update(memory_states, token_stream, token_delta)
        memory_states = memory_states + self.attn(self.ln_1(memory_states))
        memory_states = memory_states + self.mlp(self.ln_2(memory_states))
        return memory_states

class MemoryUpdateTransformer(MultiPassTransformer):
    """Recurrent model that treats shifted memory as the state being updated by tokens."""
    block_cls = MemoryUpdateBlock

    def __init__(self, config):
        super().__init__(config)
        self.mem_in_ln = LayerNorm(config.n_embd)
        self.token_in_ln = LayerNorm(config.n_embd)
        self.token_to_memory = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self._finish_initialization()
        self._init_token_to_memory_weights()
        self._init_gate_biases()

    def _init_token_to_memory_weights(self):
        with torch.no_grad():
            n_embd = self.config.n_embd
            self.token_to_memory.weight.copy_(
                torch.eye(
                    n_embd,
                    device=self.token_to_memory.weight.device,
                    dtype=self.token_to_memory.weight.dtype,
                )
            )

    def _init_gate_biases(self):
        for block in self.transformer.h:
            block.set_gate_bias(self.config.memory_gate_bias)

    def _seed_memory_stream(self, token_stream, memory_tape):
        if token_stream.shape != memory_tape.shape:
            raise ValueError("token_stream and memory_tape must have the same shape for memory updates")
        memory_base = self.mem_in_ln(memory_tape)
        token_seed = self.token_to_memory(self.token_in_ln(token_stream))
        return memory_base + token_seed

    def _run_full_pass(self, token_stream, memory_tape):
        memory_states = self._seed_memory_stream(token_stream, memory_tape)
        for block in self.transformer.h:
            memory_states = block(memory_states, token_stream)
        return memory_states

@dataclass
class MemoryUpdateConfig(MultiPassConfig):
    memory_gate_bias: float = -1.0
    use_memory_gate: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryUpdateConfig":
        return cls(
            block_size=d["block_size"],
            vocab_size=d["vocab_size"],
            n_layer=d["n_layer"],
            n_head=d["n_head"],
            n_embd=d["n_embd"],
            n_pass=d["n_pass"],
            memory_gate_bias=d.get("memory_gate_bias", -1.0),
            use_memory_gate=d.get("use_memory_gate", True),
        )
