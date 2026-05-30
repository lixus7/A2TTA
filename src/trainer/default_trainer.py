import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import networkx as nx
import torch.nn.functional as func
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch

from src.model.ewc import EWC
from torch_geometric.loader import DataLoader
from dataer.SpatioTemporalDataset import SpatioTemporalDataset
from utils.metric import cal_metric, masked_mae_np
from utils.common_tools import mkdirs, load_best_model

# ---------------------------------------------------------------------------
# Ported from ST-TTC (https://github.com/Onedean/ST-TTC, arxiv 2506.00635)
#   continual_learning_setting/src/trainer/default_trainer.py :: FRPlusModule
# Spectral-domain phase/amplitude calibrator. Consumes per-year predictions
# [B,1,N,T] and emits bias-corrected predictions of the same shape. Trained
# online from a streaming queue inside test_model_with_ttc.
# ---------------------------------------------------------------------------
class FRPlusModule(nn.Module):
    def __init__(self, num_nodes, freq_bins, groups=4):
        super().__init__()
        self.groups = groups
        self.group_size = max(1, freq_bins // groups)
        self.lambda_amp = nn.Parameter(torch.zeros(groups, num_nodes, 1))
        self.lambda_phi = nn.Parameter(torch.zeros(groups, num_nodes, 1))

    def forward(self, y_pred):
        B, C, N, T = y_pred.shape
        y = y_pred[:, 0]
        Yf = torch.fft.rfft(y, dim=-1)
        A = torch.abs(Yf)
        P = torch.angle(Yf)

        Yf_corr = torch.zeros_like(Yf)
        for g in range(self.groups):
            start = g * self.group_size
            end = T // 2 + 1 if g == self.groups - 1 else (g + 1) * self.group_size
            lam_a = self.lambda_amp[g].unsqueeze(0)
            lam_p = self.lambda_phi[g].unsqueeze(0)
            A_g = A[:, :, start:end] * (1 + lam_a)
            P_g = P[:, :, start:end] + lam_p
            Yf_corr[:, :, start:end] = A_g * torch.exp(1j * P_g)

        y_time = torch.fft.irfft(Yf_corr, n=T, dim=-1)
        return y_time.unsqueeze(1)


def train(inputs, args):
    path = osp.join(args.path, str(args.year))  # Define the current year model save path
    mkdirs(path)
    
    # Setting the loss function
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss
    elif args.loss == "mae":
        # L1/MAE — gradients scale linearly with error, not quadratically.
        # On raw-scale traffic flow (values 0-1000) this avoids the
        # exploding-gradient basin that stalls MSE+lr=0.03 for STID on
        # PEMS03/04/05/06/08. Opt-in via conf "loss": "mae".
        lossfunc = func.l1_loss
    
    # Dataset definition
    if args.strategy == 'incremental' and args.year > args.begin_year:
        # Incremental Policy Data Loader
        train_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()], y=inputs["train_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()], y=inputs["val_y"][:, :, args.subgraph.numpy()], \
            edge_index="", mode="subgraph"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        # Construct the adjacency matrix of the subgraph
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)  # Convert to adjacency matrix
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)  # Normalized adjacency matrix
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        # Common Data Loader
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=32)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
        vars(args)["sub_adj"] = vars(args)["adj"]  # Use the adjacency matrix of the entire graph
    
    # Test Data Loader
    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
    
    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model definition
    if args.init == True and args.year > args.begin_year:
        gnn_model, _ = load_best_model(args)  # If it is not the first year, load the optimal model
        if args.ewc:  # If you use the ewc strategy, use the ewc model
            args.logger.info("[*] EWC! lambda {:.6f}".format(args.ewc_lambda))  # Record EWC related parameters
            model = EWC(gnn_model, args.adj, args.ewc_lambda, args.ewc_strategy)  # Initialize the EWC model
            ewc_loader = DataLoader(SpatioTemporalDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            model.register_ewc_params(ewc_loader, lossfunc, args.device)  # Register EWC parameters
        else:
            model = gnn_model  # Otherwise, use the best model loaded
        
        if args.method == 'EAC':
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False
        
        if args.method == 'EAC':
            model.expand_adaptive_params(args.graph_size)
        
        if args.method == 'Universal' and args.use_eac == True:
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False
        
        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
        
    else:
        gnn_model = args.methods[args.method](args).to(args.device)  # If it is the first year, use the base model
        model = gnn_model
        if args.method == 'EAC':
            model.expand_adaptive_params(args.graph_size)
        
        if args.method == 'Universal' and args.use_eac == True:
            model.expand_adaptive_params(args.graph_size)
    
    if hasattr(model, 'count_parameters'):
        model.count_parameters()
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name} | Requires Grad: {param.requires_grad}")
    
    
    # Model Optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Optional linear LR warmup. Opt-in via conf "warmup_epochs" > 0.
    # Same motivation as the MAE-loss option: STID on raw-scale flow with
    # lr=0.03 + AdamW gets blown out in the first few steps for some
    # (seed, year) combos; a 5-epoch ramp from 0 -> lr fixes most of them.
    warmup_epochs = int(getattr(args, "warmup_epochs", 0) or 0)
    if warmup_epochs > 0:
        def _warmup_lambda(ep):
            return min(1.0, (ep + 1) / warmup_epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, _warmup_lambda)
    else:
        scheduler = None

    # Optional gradient-norm clipping. Opt-in via conf "grad_clip" > 0.
    grad_clip = float(getattr(args, "grad_clip", 0.0) or 0.0)


    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []
    
    for epoch in range(args.epoch):
        
        start_time = datetime.now()
        
        # Training the model
        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)
            
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred, _ = to_dense_batch(pred, batch=data.batch)  # to_dense_batch is used to convert a batch of sparse adjacency matrices into a batch of dense adjacency matrices
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]  # Slice according to the mapping to obtain the prediction and true value of the change node
                data.y = data.y[:, args.mapping, :]
            
            loss = lossfunc(data.y, pred, reduction="mean")
            
            if args.ewc and args.year > args.begin_year:
                loss += model.compute_consolidation_loss()  # Calculate and add ewc loss if necessary
            
            # NaN / Inf guard: some years in xxltrafficdata have missing sensor values.
            # If upstream preprocessing hasn't fully cleaned them, a single bad batch
            # would otherwise poison the optimizer and make every subsequent year NaN.
            if not torch.isfinite(loss):
                args.logger.warning(
                    f"[NaN-guard] year={args.year} epoch={epoch} batch={batch_idx}: "
                    f"non-finite loss={loss.item()}, "
                    f"x_nan={torch.isnan(data.x).any().item()} "
                    f"y_nan={torch.isnan(data.y).any().item()} "
                    f"pred_nan={torch.isnan(pred).any().item()} — skipping batch"
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            
            training_loss += float(loss)
            cn += 1

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        # Guard against the degenerate case where every batch in this epoch was skipped
        # by the NaN-guard above (shouldn't happen once upstream data is clean).
        training_loss = training_loss / cn if cn > 0 else float("nan")
        
        # Validate the model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                pred = model(data, args.sub_adj)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]
                
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                if not np.isfinite(loss):
                    continue
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss / cn) if cn > 0 else float("inf")
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        
        # Early Stopping Strategy
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            if args.ewc:
                torch.save({'model_state_dict': gnn_model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
            else:
                torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break
        
    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
        
    if args.ewc:
        best_model = args.methods[args.method](args)
    else:
        best_model = model
    
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    # Test the Model — dispatch ST-TTC overlay if configured
    if getattr(args, "use_ttc", 0):
        test_model_with_ttc(best_model, args, test_loader, True)
    else:
        test_model(best_model, args, test_loader, True)
    args.result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred = model(data, args.adj)
            
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
                        
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args)



def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)


# ---------------------------------------------------------------------------
# Ported from ST-TTC (https://github.com/Onedean/ST-TTC, arxiv 2506.00635)
#   continual_learning_setting/src/trainer/default_trainer.py :: test_model_with_ttc
# Differences vs upstream:
#   * Queue keeps each sparse Batch's own .batch index, so the online-update
#     step uses x_o.batch — upstream incorrectly reuses the current batch's
#     .batch and silently mis-shapes yb_o whenever batch sizes differ at the
#     tail of the year.
#   * NaN guard skips bad batches instead of poisoning the calibrator.
#   * ttc_lr / ttc_groups read from args so configs can tune per dataset.
# ---------------------------------------------------------------------------
def test_model_with_ttc(model, args, testset, pin_memory):
    import queue as _queue

    model.eval()

    T = args.y_len
    M = T // 2 + 1
    groups = int(getattr(args, "ttc_groups", 4))
    ttc_lr = float(getattr(args, "ttc_lr", 1e-4))
    # Two "gentle" knobs to stop the calibrator from over-correcting (λ grew to
    # ~15 with lr=3e-3 + unbounded cross-year carry, hurting 7/9 PEMS sets):
    #   ttc_carry_over (default 1): if 0, FRP λ is reset to identity each year
    #       (still many updates/year thanks to ttc_test_batch_size) — variant 2.
    #   ttc_lambda_clip (default 0=off): hard-clamp |λ_amp|,|λ_phi| ≤ clip after
    #       every Adam step so the multiplicative amplitude gain (1+λ) stays
    #       bounded — variant 1 (use a small value, e.g. 1.0, with a small lr).
    ttc_carry_over = int(getattr(args, "ttc_carry_over", 1))
    lam_clip = float(getattr(args, "ttc_lambda_clip", 0.0))

    # Optional: rebuild the test loader with a smaller batch_size so that the
    # online queue-full trigger fires many more times per year. Upstream's
    # default uses args.batch_size (e.g. 128), which on these PEMS years yields
    # ~15 test batches → only ~3 update steps after the T=12 warm-up. Setting
    # args.ttc_test_batch_size to e.g. 32 multiplies the per-year update count
    # by ~4× without changing the metric.
    ttc_bs = int(getattr(args, "ttc_test_batch_size", 0))
    if ttc_bs > 0 and ttc_bs != getattr(testset, "batch_size", 0):
        from torch_geometric.loader import DataLoader as _PyGDL
        testset = _PyGDL(
            testset.dataset,
            batch_size=ttc_bs,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=min(int(getattr(testset, "num_workers", 8)), 8),
        )

    FRP = FRPlusModule(args.graph_size, M, groups).to(args.device)

    # Cross-year FRP carry-over. Without this, FRP is re-init'd to λ=0 each
    # year and the ~3 update steps per year never escape identity. With
    # carry-over, previous-year's λ initializes this year's FRP, so the 25
    # years across the PEMS sweep accumulate ~75 useful Adam steps.
    # PEMS graphs grow incrementally (existing node indices stay stable, new
    # sensors append at the end), so we copy [:, :prev_N] from the saved
    # state and leave new-node rows at zero (identity init for new sensors).
    prev_state = getattr(args, "_ttc_state", None)
    if (
        ttc_carry_over
        and prev_state is not None
        and prev_state.get("groups") == groups
        and prev_state.get("M") == M
    ):
        prev_amp = prev_state["lambda_amp"]
        prev_phi = prev_state["lambda_phi"]
        shared = min(prev_amp.size(1), FRP.lambda_amp.size(1))
        with torch.no_grad():
            FRP.lambda_amp.data[:, :shared].copy_(prev_amp[:, :shared].to(args.device))
            FRP.lambda_phi.data[:, :shared].copy_(prev_phi[:, :shared].to(args.device))

    ttc_optim = optim.Adam(FRP.parameters(), lr=ttc_lr)
    q = _queue.Queue(maxsize=T)

    # Diagnostics: confirm the calibrator actually moves. With lr that's too
    # small (e.g. the previous default 1e-4 on this codebase), λ stays at ~0
    # and FRP ≡ identity, so STTTC degenerates to plain retrain.
    n_updates = 0
    abs_diff_sum = 0.0
    abs_diff_count = 0

    pred_ = []
    truth_ = []

    for data in tqdm(testset):
        data = data.to(args.device, non_blocking=pin_memory)

        with torch.no_grad():
            pred = model(data, args.adj)
        pred, _ = to_dense_batch(pred, batch=data.batch)
        y_dense, _ = to_dense_batch(data.y, batch=data.batch)

        FRP.eval()
        with torch.no_grad():
            pred_corr = FRP(pred.unsqueeze(1)).squeeze(1)
            abs_diff_sum += float((pred_corr - pred).abs().mean())
            abs_diff_count += 1

        pred_.append(pred_corr.cpu().data.numpy())
        truth_.append(y_dense.cpu().data.numpy())

        q.put((data, y_dense))
        if q.full():
            x_o, y_o = q.get()
            with torch.no_grad():
                yb_o_sparse = model(x_o, args.adj)
            yb_o, _ = to_dense_batch(yb_o_sparse, batch=x_o.batch)

            FRP.train()
            yc_o = FRP(yb_o.unsqueeze(1)).squeeze(1)
            loss = func.mse_loss(yc_o, y_o)
            if not torch.isfinite(loss):
                args.logger.warning(
                    f"[TTC NaN-guard] year={args.year}: non-finite calibrator loss, skipping update"
                )
                ttc_optim.zero_grad(set_to_none=True)
                FRP.eval()
                continue
            ttc_optim.zero_grad()
            loss.backward()
            ttc_optim.step()
            if lam_clip > 0:
                with torch.no_grad():
                    FRP.lambda_amp.clamp_(-lam_clip, lam_clip)
                    FRP.lambda_phi.clamp_(-lam_clip, lam_clip)
            n_updates += 1
            FRP.eval()

    lam_a_l2 = float(FRP.lambda_amp.detach().norm())
    lam_p_l2 = float(FRP.lambda_phi.detach().norm())
    avg_corr = abs_diff_sum / max(abs_diff_count, 1)
    args.logger.info(
        f"[TTC] year={args.year} updates={n_updates} "
        f"|λ_amp|={lam_a_l2:.4g} |λ_phi|={lam_p_l2:.4g} "
        f"mean|FRP(y)-y|={avg_corr:.4g} "
        f"(ttc_lr={ttc_lr} carry={ttc_carry_over} clip={lam_clip})"
    )

    # Persist FRP state so next year's call warm-starts from this year's λ.
    args._ttc_state = {
        "groups": groups,
        "M": M,
        "lambda_amp": FRP.lambda_amp.detach().cpu().clone(),
        "lambda_phi": FRP.lambda_phi.detach().cpu().clone(),
    }

    pred_ = np.concatenate(pred_, 0)
    truth_ = np.concatenate(truth_, 0)
    cal_metric(truth_, pred_, args)