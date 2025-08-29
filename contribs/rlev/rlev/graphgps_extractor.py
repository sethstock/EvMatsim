import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool


def _to_device(arr, device: torch.device):
    return torch.as_tensor(arr, device=device)


def _build_fixed_subgraph(edge_index_2E: torch.Tensor, K: int, N: int) -> tuple[torch.Tensor, torch.Tensor]:
    # degree-based fixed node subset (capture-safe; no host sync)
    deg = torch.bincount(edge_index_2E.view(-1), minlength=N)
    K = min(K, N)
    sel_idx = torch.topk(deg, k=K, largest=True, sorted=True).indices  # (K,)
    mask = torch.zeros(N, dtype=torch.bool, device=edge_index_2E.device); mask[sel_idx] = True
    u, v = edge_index_2E[0], edge_index_2E[1]
    keep = mask[u] & mask[v]
    ei = edge_index_2E[:, keep]  # (2, E_sel) in original ids
    remap = -torch.ones(N, dtype=torch.long, device=edge_index_2E.device)
    remap[sel_idx] = torch.arange(sel_idx.numel(), device=edge_index_2E.device)
    ei_sel = remap[ei]  # reindexed to [0..K-1]
    return sel_idx, ei_sel


class LocalGlobalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, attn_dropout: float):
        super().__init__()
        mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.local = GINConv(nn=mlp, train_eps=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x_flat: torch.Tensor, edge_index: torch.Tensor, B: int, K: int) -> torch.Tensor:
        # x_flat: (B*K, D)
        x = self.norm1(x_flat)
        h = self.local(x, edge_index)                 # (B*K, D)
        x = x + self.drop(h)                          # residual after local
        xd = x.view(B, K, -1)                         # (B, K, D) dense tokens for attention
        y, _ = self.attn(xd, xd, xd, need_weights=False)  # (B, K, D)
        y = y.reshape(B * K, -1)
        x = self.norm2(x + self.drop(y))              # residual after global
        return x


class GraphGPSExtractor(BaseFeaturesExtractor):
    """
    Capture-safe GraphGPS-style extractor:
      - fixed K nodes per graph (constant shapes),
      - local GIN on (B*K,D) with batched edges,
      - global MHA on (B,K,D) without to_dense_batch,
      - mixed precision + CUDA Graphs (optional).
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.2,
        fixed_k: int = 128,
        use_amp: bool = True,
        use_cudagraphs: bool = True,
    ):
        super().__init__(observation_space, features_dim)
        self.embed_dim = features_dim
        self.fixed_k = int(fixed_k)
        self.use_amp = bool(use_amp)
        self._want_cudagraphs = bool(use_cudagraphs)

        in_dim = int(observation_space["nodes"].shape[-1])  # type: ignore[attr-defined]
        self.in_proj = nn.Identity() if in_dim == self.embed_dim else nn.Linear(in_dim, self.embed_dim)
        self.blocks = nn.ModuleList([
            LocalGlobalBlock(self.embed_dim, heads, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(self.embed_dim)

        # Templates & capture state
        self._sel_idx: Optional[torch.Tensor] = None      # (K,)
        self._ei_single: Optional[torch.Tensor] = None    # (2, E_sel)
        self._batched_ei: Optional[torch.Tensor] = None   # (2, B*E_sel)
        self._batch_vec: Optional[torch.Tensor] = None    # (B*K,)

        self._static_nodes: Optional[torch.Tensor] = None # (B, K, F)
        self._static_out: Optional[torch.Tensor] = None   # (B, D)

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._captured: bool = False
        self._B: Optional[int] = None
        self._F: Optional[int] = None

    # ---------- fixed templates ----------
    def _ensure_templates(self, obs: Dict[str, Any], device: torch.device):
        x = _to_device(obs["nodes"], device)       # (B,N,F) or (N,F)
        ei = _to_device(obs["edge_index"], device) # (B,2,E) or (2,E)

        if x.dim() == 2:
            B, N, F = 1, x.size(0), x.size(1); ei0 = ei
        else:
            B, N, F = x.size(0), x.size(1), x.size(2); ei0 = ei[0]

        if self._sel_idx is not None and self._B == B and self._F == F:
            return

        sel_idx, ei_sel = _build_fixed_subgraph(ei0.long(), self.fixed_k, N)  # (K,), (2,E_sel)
        K = sel_idx.numel()
        self._sel_idx, self._ei_single = sel_idx, ei_sel

        E_sel = ei_sel.size(1)
        offsets = (torch.arange(B, device=device) * K).view(1, B, 1)    # (1,B,1)
        ei_b = ei_sel.unsqueeze(1).expand(2, B, E_sel) + offsets        # (2,B,E_sel)
        self._batched_ei = ei_b.reshape(2, B * E_sel).contiguous()      # (2, B*E_sel)
        self._batch_vec = torch.repeat_interleave(torch.arange(B, device=device), K)  # (B*K,)

        # static buffers (AMP-aware dtype)
        dtype_nodes = torch.float16 if (device.type == "cuda" and self.use_amp) else torch.float32
        self._static_nodes = torch.empty((B, K, F), dtype=dtype_nodes, device=device)
        self._static_out = torch.empty((B, self.embed_dim), dtype=torch.float32, device=device)

        self._B, self._F = B, F
        self._captured = False

    def _assemble_data_from_static(self) -> Data:
        x_flat = self._static_nodes.view(self._B * self._sel_idx.numel(), self._F)  # type: ignore[arg-type]
        return Data(x=x_flat, edge_index=self._batched_ei, batch=self._batch_vec)

    # ---------- forward (no capture) ----------
    def _stack_forward(self, data: Data) -> torch.Tensor:
    # Mixed precision for the whole stack (no deprecation warning)
        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp and data.x.is_cuda):
            x = self.in_proj(data.x)  # (B*K, D)
            B = self._B
            K = self._sel_idx.numel()

        # Local+global blocks (capture-safe)
            for block in self.blocks:
                x = block(x, data.edge_index, B=B, K=K)  # (B*K, D)

        # <<< IMPORTANT: capture-safe pooling (no PyG scatter, no host sync) >>>
        # We know shapes are fixed: x is (B*K, D). Just reshape and average tokens.
            g = x.view(B, K, -1).mean(dim=1).contiguous()  # (B, D)
            g = self.out_norm(g)

        return g.float()


    # ---------- capture ----------
    @torch.no_grad()
    def _maybe_capture(self, obs: Dict[str, Any], device: torch.device):
        if not self._want_cudagraphs or self._captured or device.type != "cuda":
            return

        self._ensure_templates(obs, device)

        # Warmup on default stream
        x = _to_device(obs["nodes"], device)
        if x.dim() == 2: x = x.unsqueeze(0)  # (1,N,F)
        sel = self._sel_idx
        self._static_nodes.copy_(x[:, sel, :].to(self._static_nodes.dtype))
        _ = self._stack_forward(self._assemble_data_from_static())
        torch.cuda.synchronize(device)

        # Capture on a non-default stream
        self._capture_stream = torch.cuda.Stream(device=device)
        g = torch.cuda.CUDAGraph()
        torch.cuda.current_stream(device).wait_stream(self._capture_stream)
        with torch.cuda.stream(self._capture_stream):
            with torch.cuda.graph(g):
                out = self._stack_forward(self._assemble_data_from_static())
                self._static_out.copy_(out)
        torch.cuda.current_stream(device).wait_stream(self._capture_stream)

        self._graph = g
        self._captured = True

    # ---------- public forward ----------
    def forward(self, obs: Dict[str, Any]) -> torch.Tensor:
        dev = next(self.parameters()).device
        self._ensure_templates(obs, dev)

        x = _to_device(obs["nodes"], dev)
        if x.dim() == 2: x = x.unsqueeze(0)  # (1,N,F)
        sel = self._sel_idx
        self._static_nodes.copy_(x[:, sel, :].to(self._static_nodes.dtype))

        if self._want_cudagraphs and dev.type == "cuda" and not self._captured:
            self._maybe_capture(obs, dev)

        if self._want_cudagraphs and self._captured and self._graph is not None:
            self._graph.replay()
            return self._static_out
        else:
            return self._stack_forward(self._assemble_data_from_static())
