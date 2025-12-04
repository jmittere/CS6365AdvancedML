"""
LightGCN Runner: Optimized for large-scale datasets with memory-efficient evaluation.

Key Optimizations:
1) AMP disabled for sparse ops (keep FP32 for torch.sparse.mm)
2) Chunk-based evaluation to prevent OOM
3) Efficient Top-K without full score materialization
4) Gradient accumulation option
5) K negatives/pos and edge dropout
"""

import os, math, json, argparse, time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix

# ----------------------------- Utils -----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def now_ts():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

# ----------------------------- I/O -----------------------------
def read_split(p: str) -> pd.DataFrame:
    if not os.path.exists(p): raise FileNotFoundError(p)
    df = pd.read_csv(p, usecols=["user_id_int","item_id_int"]).astype(np.int64)
    if df.isna().any().any(): raise ValueError(f"NaNs in {p}")
    return df

def infer_counts(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[int,int]:
    num_users = int(max(train_df.user_id_int.max(), val_df.user_id_int.max(), test_df.user_id_int.max())) + 1
    num_items = int(max(train_df.item_id_int.max(), val_df.item_id_int.max(), test_df.item_id_int.max())) + 1
    return num_users, num_items

def build_user_pos_dict(df: pd.DataFrame) -> Dict[int, Set[int]]:
    d = defaultdict(set)
    for u,i in zip(df.user_id_int.values, df.item_id_int.values):
        d[int(u)].add(int(i))
    return d

def build_norm_adj(num_users: int,
                   num_items: int,
                   train_df: pd.DataFrame,
                   device,
                   edge_dropout: float = 0.0) -> torch.sparse.FloatTensor:
    """
    Build Â = D^{-1/2} A D^{-1/2} on CPU, then move once to device.
    If edge_dropout>0, randomly drop that fraction of (u,i) edges BEFORE symmetrization.
    """
    u = train_df.user_id_int.values.astype(np.int64)
    i = train_df.item_id_int.values.astype(np.int64)

    if edge_dropout > 0.0:
        keep_mask = (np.random.rand(u.shape[0]) > edge_dropout)
        u = u[keep_mask]; i = i[keep_mask]

    n_nodes = num_users + num_items
    rows = np.concatenate([u, i + num_users])
    cols = np.concatenate([i + num_users, u])
    data = np.ones(rows.shape[0], dtype=np.float32)

    A = coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr().tocoo()
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    deg[deg == 0] = 1.0
    d_inv_sqrt = np.power(deg, -0.5)

    norm_data = d_inv_sqrt[A.row] * A.data * d_inv_sqrt[A.col]
    A_norm = coo_matrix((norm_data, (A.row, A.col)), shape=A.shape)

    idx = torch.tensor(np.vstack([A_norm.row, A_norm.col]), dtype=torch.long)
    val = torch.tensor(A_norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=A_norm.shape).coalesce().to(device)

# ----------------------------- Model -----------------------------
class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int, num_layers: int, A_norm: torch.sparse.FloatTensor):
        super().__init__()
        self.num_users, self.num_items, self.num_layers = num_users, num_items, num_layers
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.register_buffer("A_norm", A_norm)
        self._cache_u = None; self._cache_i = None

    def propagate(self, use_cache: bool = False):
        if use_cache and self._cache_u is not None:
            return self._cache_u, self._cache_i
        all_e = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        xs = [all_e]; x = all_e
        for _ in range(self.num_layers):
            x = torch.sparse.mm(self.A_norm, x)  # keep FP32
            xs.append(x)
        out = torch.mean(torch.stack(xs, dim=0), dim=0)
        self._cache_u, self._cache_i = out[:self.num_users], out[self.num_users:]
        return self._cache_u, self._cache_i

    def clear_cache(self):
        self._cache_u = None; self._cache_i = None

# ----------------------------- Loss & Batching -----------------------------
def bpr_loss_multi(u_emb: torch.Tensor,
                   pos_emb: torch.Tensor,
                   neg_emb: torch.Tensor,
                   reg_lambda: float = 0.0) -> torch.Tensor:
    """
    u_emb:   [B, d]
    pos_emb: [B, d]
    neg_emb: [B, K, d]
    """
    # [B, K]
    score_pos = (u_emb * pos_emb).sum(dim=-1, keepdim=True)           # [B,1]
    score_neg = torch.einsum('bd,bkd->bk', u_emb, neg_emb)            # [B,K]
    x = score_pos - score_neg                                         # [B,K]
    loss = -F.logsigmoid(x).mean()

    if reg_lambda > 0.0:
        reg = (u_emb.norm(dim=1).pow(2) + pos_emb.norm(dim=1).pow(2) + neg_emb.norm(dim=2).pow(2).mean(dim=1)).mean()
        loss = loss + reg_lambda * reg
    return loss

def iter_batches(pairs: np.ndarray, batch_size: int):
    idx = np.arange(pairs.shape[0]); np.random.shuffle(idx)
    for s in range(0, idx.size, batch_size):
        b = idx[s:s+batch_size]
        yield pairs[b, 0], pairs[b, 1]

class NegativeSampler:
    def __init__(self, num_items: int, user_pos: Dict[int, Set[int]]):
        self.num_items, self.user_pos = num_items, user_pos

    def sample_k(self, users: np.ndarray, k: int) -> np.ndarray:
        """
        Return negatives shape [B, K] with rejection sampling per user.
        """
        B = len(users)
        negs = np.empty((B, k), dtype=np.int64)
        for t, u in enumerate(users):
            seen = self.user_pos.get(int(u), set())
            c = 0
            while c < k:
                j = np.random.randint(0, self.num_items)
                if j not in seen:
                    negs[t, c] = j
                    c += 1
        return negs

# ----------------------------- Memory-Efficient Evaluation -----------------------------
@torch.no_grad()
def evaluate_chunked(model: nn.Module,
                     split_df: pd.DataFrame,
                     user_train_pos: Dict[int, Set[int]],
                     num_items: int,
                     pop: np.ndarray,
                     hist: Dict[int, int],
                     device,
                     k_list=(10,20),
                     user_chunk_size: int = 1024,
                     item_chunk_size: int = 10000):
    model.eval()
    U, I = model.propagate(use_cache=True)
    Kmax = min(max(k_list), num_items)

    acc = {k: {'recall': [], 'ndcg': [], 'hit': []} for k in k_list}
    rec_lists = {k: [] for k in k_list}
    buckets = {"3-5": [], "6-10": [], ">10": []}

    users_np = split_df.user_id_int.values
    gts_np = split_df.item_id_int.values

    print(f"Evaluating {len(split_df)} users in chunks of {user_chunk_size}...")

    for user_start in range(0, len(split_df), user_chunk_size):
        user_end = min(user_start + user_chunk_size, len(split_df))
        chunk_users = users_np[user_start:user_end]
        chunk_gts = gts_np[user_start:user_end]

        U_chunk = U[torch.tensor(chunk_users, device=device, dtype=torch.long)]
        chunk_size = len(chunk_users)

        top_k_scores = torch.full((chunk_size, Kmax), -1e9, device=device)
        top_k_items  = torch.zeros((chunk_size, Kmax), device=device, dtype=torch.long)

        for item_start in range(0, num_items, item_chunk_size):
            item_end = min(item_start + item_chunk_size, num_items)
            I_chunk = I[item_start:item_end]                                   # [Ic, d]
            scores  = torch.matmul(U_chunk, I_chunk.t())                       # [B, Ic]

            # Mask training items
            for local_idx, u in enumerate(chunk_users):
                seen = user_train_pos.get(int(u), None)
                if seen:
                    # mask only those inside current chunk
                    for j in seen:
                        if item_start <= j < item_end:
                            scores[local_idx, j - item_start] = -1e9

            item_indices = torch.arange(item_start, item_end, device=device)
            combined_scores = torch.cat([top_k_scores, scores], dim=1)
            combined_items  = torch.cat([top_k_items, item_indices.unsqueeze(0).expand(chunk_size, -1)], dim=1)
            top_k_scores, topk_idx = torch.topk(combined_scores, k=Kmax, dim=1)
            top_k_items = torch.gather(combined_items, 1, topk_idx)

        # Metrics
        for local_idx, (u, gt) in enumerate(zip(chunk_users, chunk_gts)):
            gt = int(gt)
            topi = top_k_items[local_idx]

            for k in k_list:
                tk = topi[:k]
                hit = float((tk == gt).any().item())
                pos = (tk == gt).nonzero(as_tuple=False)
                ndc = 0.0 if pos.numel() == 0 else 1.0 / math.log2(int(pos[0].item()) + 2)
                acc[k]['hit'].append(hit)
                acc[k]['recall'].append(hit)   # one GT -> Recall@k == Hit@k
                acc[k]['ndcg'].append(ndc)
                rec_lists[k].extend(tk.tolist())

            # Cold buckets at @20
            n = hist.get(int(u), 0)
            b = "3-5" if 3<=n<=5 else ("6-10" if 6<=n<=10 else ">10")
            tk20 = topi[:20] if 20 <= Kmax else topi
            buckets[b].append(float((tk20 == gt).any().item()))

        if (user_end) % 10000 == 0 or user_end == len(split_df):
            print(f"  Processed {user_end}/{len(split_df)} users | GPU mem: {get_gpu_memory():.2f} GB")

    per_k = {k: {
        'recall': float(np.mean(acc[k]['recall'])),
        'ndcg': float(np.mean(acc[k]['ndcg'])),
        'hitrate': float(np.mean(acc[k]['hit']))
    } for k in k_list}

    top_key = 20 if 20 in rec_lists else max(k_list)
    rec20 = np.array(rec_lists[top_key], dtype=np.int64)
    coverage20 = len(set(rec20.tolist())) / float(num_items) if num_items>0 else 0.0
    avgpop20   = float(pop[rec20].mean()) if rec20.size>0 else 0.0
    cold = {b: (float(np.mean(v)) if len(v) else 0.0) for b,v in buckets.items()}

    return {'per_k': per_k, 'coverage@20': coverage20, 'avgpop@20': avgpop20, 'cold_user_recall@20': cold}

# ----------------------------- Checkpointing -----------------------------
def save_checkpoint(epoch, model, optimizer, best_val, extra, path):
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_val": best_val,
        "extra": extra,
    }, path)

def rotate_checkpoints(dirpath, keep_last_n=5, prefix="ckpt_epoch_"):
    ensure_dir(dirpath)
    files = sorted([f for f in os.listdir(dirpath) if f.startswith(prefix)])
    for f in files[:-keep_last_n]:
        try: os.remove(os.path.join(dirpath, f))
        except OSError: pass

# ----------------------------- Runner -----------------------------
def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.results_dir)

    # Load splits
    train_df = read_split(args.train_csv)
    val_df   = read_split(args.val_csv)
    test_df  = read_split(args.test_csv)
    num_users, num_items = infer_counts(train_df, val_df, test_df)
    print(f"Users={num_users:,} | Items={num_items:,} | Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    print(f"Sparsity(train): {1 - len(train_df)/(num_users*num_items):.6f}")

    # Graph & helpers
    print("Building normalized adjacency matrix...")
    A_norm = build_norm_adj(num_users, num_items, train_df, device, edge_dropout=args.edge_dropout)
    user_pos   = build_user_pos_dict(train_df)
    train_pairs= np.stack([train_df.user_id_int.values, train_df.item_id_int.values], axis=1).astype(np.int64)
    neg_sampler= NegativeSampler(num_items, user_pos)
    pop        = np.bincount(train_df.item_id_int.values, minlength=num_items)
    hist       = train_df.groupby("user_id_int").size().to_dict()

    # Model & optim
    model = LightGCN(num_users, num_items, args.embed_dim, args.layers, A_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    start_epoch, best_val = 1, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("best_val", -1.0)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[RESUME] from {args.resume} -> start_epoch={start_epoch}, best_val={best_val:.4f}")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch={args.batch_size} | GradAccum={args.grad_accum_steps} | layers={args.layers} | dim={args.embed_dim}")
    print("NOTE: AMP disabled for sparse ops (LightGCN propagation stays FP32).")

    # Train loop
    best_state = None
    curve_rows = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); model.clear_cache()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (users_np, pos_np) in enumerate(iter_batches(train_pairs, args.batch_size)):
            neg_np = neg_sampler.sample_k(users_np, args.negs_per_pos)  # [B, K]

            users = torch.as_tensor(users_np, device=device, dtype=torch.long)
            pos   = torch.as_tensor(pos_np,   device=device, dtype=torch.long)
            neg   = torch.as_tensor(neg_np,   device=device, dtype=torch.long)   # [B, K]

            U, I = model.propagate()
            u_emb = U[users]                 # [B, d]
            i_emb = I[pos]                   # [B, d]
            j_emb = I[neg]                   # [B, K, d]

            loss = bpr_loss_multi(u_emb, i_emb, j_emb, reg_lambda=args.emb_l2)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * args.grad_accum_steps
            n_batches += 1

        # Final step if remainder grads
        optimizer.step(); optimizer.zero_grad(set_to_none=True)

        # Validation
        model.clear_cache(); model.eval(); _ = model.propagate(use_cache=True)
        val = evaluate_chunked(model, val_df, user_pos, num_items, pop, hist, device,
                               k_list=tuple(args.eval_k),
                               user_chunk_size=args.eval_user_chunk,
                               item_chunk_size=args.eval_item_chunk)
        pk = val['per_k']
        avg_loss = epoch_loss / max(1, n_batches)
        val_line = {f"R@{k}": pk[k]['recall'] for k in args.eval_k}
        val_line.update({f"N@{k}": pk[k]['ndcg'] for k in args.eval_k})
        print(f"Epoch {epoch:03d} | Loss={avg_loss:.4f} | " +
              " | ".join([f"@{k}: R={pk[k]['recall']:.4f}, NDCG={pk[k]['ndcg']:.4f}, HR={pk[k]['hitrate']:.4f}"
                          for k in args.eval_k]))
        print(f"Coverage@20={val['coverage@20']:.4f} | AvgPop@20={val['avgpop@20']:.2f} | Cold@20={val['cold_user_recall@20']}")

        # Save curve row
        row = {"epoch": epoch, "loss": avg_loss, **val_line,
               "coverage@20": val["coverage@20"], "avgpop@20": val["avgpop@20"]}
        curve_rows.append(row)

        # Early stopping metric: NDCG@maxK
        maxK = max(args.eval_k)
        score_for_early = pk[maxK]['ndcg']
        if score_for_early >= best_val:
            best_val = score_for_early
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch,
                "state_dict": best_state,
                "best_val": best_val,
                "extra": {"config": vars(args)}
            }, os.path.join(args.checkpoint_dir, "ckpt_best.pt"))
            print(f"✓ New best saved (NDCG@{maxK}={best_val:.4f})")

        # Rotate epoch checkpoints
        ep_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch_{epoch:03d}.pt")
        save_checkpoint(epoch, model, optimizer, best_val, {"val": val, "config": vars(args)}, ep_path)
        rotate_checkpoints(args.checkpoint_dir, args.keep_last, prefix="ckpt_epoch_")

        # Persist curve so far
        pd.DataFrame(curve_rows).to_csv(os.path.join(args.results_dir, "training_curve.csv"), index=False)

    # Final test with best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.clear_cache()
    print("\nFinal test evaluation:")
    test = evaluate_chunked(model, test_df, user_pos, num_items, pop, hist, device,
                            k_list=tuple(args.eval_k),
                            user_chunk_size=args.eval_user_chunk,
                            item_chunk_size=args.eval_item_chunk)
    pk = test['per_k']
    print("\n=== FINAL TEST ===")
    for k in args.eval_k:
        print(f"@{k:2d}  Recall={pk[k]['recall']:.4f}  NDCG={pk[k]['ndcg']:.4f}  HitRate={pk[k]['hitrate']:.4f}")
    print(f"Coverage@20={test['coverage@20']:.4f}  AvgPop@20={test['avgpop@20']:.2f}")
    print(f"Cold-user Recall@20: {test['cold_user_recall@20']}")

    # Save final model + metrics
    torch.save({
        "state_dict": model.state_dict(),
        "num_users": num_users, "num_items": num_items,
        "embed_dim": args.embed_dim, "num_layers": args.layers
    }, os.path.join(args.checkpoint_dir, "lightgcn_model.pt"))

    summary = {
        "timestamp": now_ts(),
        "config": vars(args),
        "final_test": test
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {os.path.join(args.checkpoint_dir, 'lightgcn_model.pt')}")
    print(f"Saved -> {os.path.join(args.results_dir, 'metrics.json')}")
    print(f"Saved -> {os.path.join(args.results_dir, 'training_curve.csv')}")

def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)

    if not args.ckpt: raise ValueError("--ckpt is required for --mode eval")

    train_df = read_split(args.train_csv)
    val_df   = read_split(args.val_csv)
    test_df  = read_split(args.test_csv)
    num_users, num_items = infer_counts(train_df, val_df, test_df)
    A_norm = build_norm_adj(num_users, num_items, train_df, device, edge_dropout=0.0)
    user_pos = build_user_pos_dict(train_df)
    pop = np.bincount(train_df.item_id_int.values, minlength=num_items)
    hist = train_df.groupby("user_id_int").size().to_dict()

    ckpt = torch.load(args.ckpt, map_location=device)
    embed_dim = ckpt.get("extra", {}).get("config", {}).get("embed_dim", args.embed_dim)
    num_layers = ckpt.get("extra", {}).get("config", {}).get("layers", args.layers)

    model = LightGCN(num_users, num_items, embed_dim, num_layers, A_norm).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval(); model.clear_cache(); _ = model.propagate(use_cache=True)

    for split_name, df in (("VAL", val_df), ("TEST", test_df)):
        print(f"\nEvaluating {split_name}:")
        out = evaluate_chunked(model, df, user_pos, num_items, pop, hist, device,
                               k_list=tuple(args.eval_k),
                               user_chunk_size=args.eval_user_chunk,
                               item_chunk_size=args.eval_item_chunk)
        pk = out['per_k']
        print(f"\n=== {split_name} ===")
        for k in args.eval_k:
            print(f"@{k:2d}  Recall={pk[k]['recall']:.4f}  NDCG={pk[k]['ndcg']:.4f}  HitRate={pk[k]['hitrate']:.4f}")
        print(f"Coverage@20={out['coverage@20']:.4f}  AvgPop@20={out['avgpop@20']:.2f}")
        print(f"Cold-user Recall@20: {out['cold_user_recall@20']}")

# ----------------------------- CLI -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","eval"], default="train")

    # Data
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)

    # Model
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)

    # Train/Eval
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--emb_l2", type=float, default=1e-4)
    ap.add_argument("--negs_per_pos", type=int, default=1, help="Number of negatives per positive")
    ap.add_argument("--edge_dropout", type=float, default=0.0, help="Fraction of edges to drop when building A_norm [0,1)")

    ap.add_argument("--eval_k", type=int, nargs="+", default=[10,20])
    ap.add_argument("--eval_user_chunk", type=int, default=1024)
    ap.add_argument("--eval_item_chunk", type=int, default=10000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Paths
    ap.add_argument("--checkpoint_dir", default="./checkpoints")
    ap.add_argument("--results_dir", default="./results")

    # Resume/Eval
    ap.add_argument("--resume", default=None)
    ap.add_argument("--keep_last", type=int, default=5)
    ap.add_argument("--ckpt", default=None)

    # Viz toggle (external script optional)
    ap.add_argument("--auto_visualize", action="store_true")
    ap.add_argument("--viz_output_dir", default="./results")

    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_eval(args)

if __name__ == "__main__":
    main()
