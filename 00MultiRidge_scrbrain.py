import argparse, os
import numpy as np
from himalaya.backend import set_backend
from himalaya.scoring import correlation_score
from sklearn.preprocessing import StandardScaler
import torch
import time

# ============辅助函数，计算权值范数并保存===========
def weight_norms(coef, norm_type="l2"):
    # filepath: /home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/codes/utils/00MultiRidge copy.py
    if isinstance(coef, torch.Tensor):
        coef = coef.detach().float().cpu().numpy()
    if norm_type == "l2":
        return np.linalg.norm(coef, ord=2, axis=1)
    if norm_type == "l1":
        return np.linalg.norm(coef, ord=1, axis=1)
    raise ValueError("norm_type must be 'l1' or 'l2'")

def save_weight_norms(out_dir, base_name, l1_vals, l2_vals):
    os.makedirs(out_dir, exist_ok=True)
    percentiles = {
        "l1": np.percentile(l1_vals, [25, 50, 75]),
        "l2": np.percentile(l2_vals, [25, 50, 75]),
    }
    np.savez(
        os.path.join(out_dir, f"{base_name}_weight_norms.npz"),
        l1_norms=l1_vals,
        l2_norms=l2_vals,
        l1_percentiles=percentiles["l1"],
        l2_percentiles=percentiles["l2"],
    )
    return percentiles

# ============ GPU-accelerated Woodbury Ridge 实现 ============
class WoodburyRidgeTorch:
    """基于 Woodbury 恒等式的 GPU 加速岭回归实现（支持 float16）"""
    
    def __init__(self, alpha=1.0, fit_intercept=True, device='cuda', dtype=torch.float16):
        """
        Arguments:
            alpha: 正则化强度 (λ)
            fit_intercept: 是否拟合截距
            device: 'cuda' 或 'cpu'
            dtype: 数据类型 (torch.float16 或 torch.float32)
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype  # ← 新增：支持自定义数据类型
        
        # 内部参数
        self.X_mean_ = None
        self.Y_mean_ = None
        self.U_ = None
        self.s_ = None
        self.Vt_ = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, Y):
        """拟合模型"""
        # 转换为指定的数据类型（float16 或 float32）
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.dtype).to(self.device)  # ← 改这里
        else:
            X = X.to(self.dtype).to(self.device)
            
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).to(self.dtype).to(self.device)  # ← 改这里
        else:
            Y = Y.to(self.dtype).to(self.device)
        
        n_samples, n_features = X.shape
        
        # 1. 中心化
        if self.fit_intercept:
            self.X_mean_ = torch.mean(X, dim=0, keepdim=True)
            self.Y_mean_ = torch.mean(Y, dim=0, keepdim=True)
            X_centered = X - self.X_mean_
            Y_centered = Y - self.Y_mean_
        else:
            X_centered = X.clone()
            Y_centered = Y.clone()
            self.X_mean_ = torch.zeros(1, n_features, device=self.device, dtype=self.dtype)
            self.Y_mean_ = torch.zeros(1, Y.shape[1], device=self.device, dtype=self.dtype)
        
        # 2. SVD 分解 (GPU 加速)
        # 注意：SVD 在 float16 下可能不稳定，建议在 float32 下计算后转回
        if self.dtype == torch.float16:
            X_centered_fp32 = X_centered.float()  # 临时转 float32
            self.U_, self.s_, self.Vt_ = torch.linalg.svd(X_centered_fp32, full_matrices=False)
            self.U_ = self.U_.half()  # 转回 float16
            self.s_ = self.s_.half()
            self.Vt_ = self.Vt_.half()
            del X_centered_fp32
        else:
            self.U_, self.s_, self.Vt_ = torch.linalg.svd(X_centered, full_matrices=False)
        
        # 3. 使用 Woodbury 公式计算系数
        s_reg = self.s_ / (self.s_**2 + self.alpha)
        UtY = self.U_.t() @ Y_centered
        s_UtY = s_reg.unsqueeze(1) * UtY
        self.coef_ = self.Vt_.t() @ s_UtY
        
        # 4. 计算截距
        if self.fit_intercept:
            self.intercept_ = self.Y_mean_ - self.X_mean_ @ self.coef_
        else:
            self.intercept_ = torch.zeros(1, Y.shape[1], device=self.device, dtype=self.dtype)
            
        return self
    
    def predict(self, X):
        """预测"""
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        # 转换为指定数据类型
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.dtype).to(self.device)  # ← 改这里
        else:
            X = X.to(self.dtype).to(self.device)
        
        Y_pred = X @ self.coef_
        
        if self.fit_intercept:
            Y_pred += self.intercept_
            
        return Y_pred
    
    def score(self, X, Y):
        """计算 R^2 分数"""
        Y_pred = self.predict(X)
        
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).to(self.dtype).to(self.device)
        else:
            Y = Y.to(self.dtype).to(self.device)
        
        ss_res = torch.sum((Y - Y_pred) ** 2)
        ss_tot = torch.sum((Y - torch.mean(Y, dim=0)) ** 2)
        
        score = 1 - (ss_res / ss_tot)
        return score.cpu().item()


class WoodburyRidgeCVTorch:
    """GPU 加速的带交叉验证的 Woodbury 岭回归（支持 float16）"""
    
    def __init__(self, alphas, fit_intercept=True, cv=5, scoring='r2', device='cuda', dtype=torch.float16):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.scoring = scoring
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype  # ← 新增
        
        self.best_alpha_ = None
        self.best_model_ = None
        self.cv_scores_ = None
        
    def fit(self, X, Y):
        """拟合模型并通过交叉验证选择最优 alpha"""
        # 转换为 PyTorch tensor
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.dtype).to(self.device)
        else:
            X = X.to(self.dtype).to(self.device)
            
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).to(self.dtype).to(self.device)
        else:
            Y = Y.to(self.dtype).to(self.device)
        
        n_samples = X.shape[0]
        n_alphas = len(self.alphas)
        
        cv_scores = np.zeros(n_alphas)
        fold_size = n_samples // self.cv
        
        print(f"\n========== Cross-Validation on {self.device} ({self.dtype}) ==========")
        
        for alpha_idx, alpha in enumerate(self.alphas):
            fold_scores = []
            
            for fold in range(self.cv):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < self.cv - 1 else n_samples
                
                val_mask = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
                val_mask[val_start:val_end] = True
                train_mask = ~val_mask
                
                X_train, X_val = X[train_mask], X[val_mask]
                Y_train, Y_val = Y[train_mask], Y[val_mask]
                
                # 训练模型
                model = WoodburyRidgeTorch(
                    alpha=alpha, 
                    fit_intercept=self.fit_intercept,
                    device=self.device,
                    dtype=self.dtype  # ← 传递数据类型
                )
                model.fit(X_train, Y_train)
                
                # 评估
                if self.scoring == 'r2':
                    score = model.score(X_val, Y_val)
                elif self.scoring == 'correlation':
                    Y_pred = model.predict(X_val)
                    corr = correlation_score(Y_val.t(), Y_pred.t())
                    if hasattr(corr, 'mean'):
                        score = corr.mean().cpu().item()
                    else:
                        score = float(np.mean(corr))
                
                fold_scores.append(score)
            
            cv_scores[alpha_idx] = np.mean(fold_scores)
            print(f"Alpha {alpha:.2e}: CV score = {cv_scores[alpha_idx]:.4f}")
            
        # 选择最优 alpha
        best_idx = np.argmax(cv_scores)
        self.best_alpha_ = self.alphas[best_idx]
        self.cv_scores_ = cv_scores
        
        print(f"\n✓ Best alpha: {self.best_alpha_:.2e}, CV score: {cv_scores[best_idx]:.4f}")
        
        # 使用最优 alpha 在全部数据上重新训练
        self.best_model_ = WoodburyRidgeTorch(
            alpha=self.best_alpha_,
            fit_intercept=self.fit_intercept,
            device=self.device,
            dtype=self.dtype  # ← 传递数据类型
        )
        self.best_model_.fit(X, Y)
        
        return self
    
    def predict(self, X):
        if self.best_model_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        return self.best_model_.predict(X)
    
    def score(self, X, Y):
        return self.best_model_.score(X, Y)

# ============ 主函数 ============
def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default='init_latent', type=str, help="Target variable")
    parser.add_argument("--subject", default='subj01', type=str, help="subject name")
    parser.add_argument("--device", default='cuda', type=str, help="Device: cuda or cpu")
    parser.add_argument("--fp16", action='store_true', help="Use float16")  # ← 新增
    
    opt = parser.parse_args()
    target = opt.target
    subject = opt.subject
    device = opt.device if torch.cuda.is_available() else 'cpu'
    
    data_type = 'ave'
    
    # 参数设置
    subj_type = 'share'
    ROI_NAME = 's0_vis'
    
    # 2. 检查 GPU 可用性
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Using device: {device}")
    
    # 设置后端
    backend = set_backend("torch" if torch.cuda.is_available() else "numpy", on_error="warn")
    print(f"Himalaya backend: {backend.name}")
    
    # 3. 根据目标变量选择正则化参数（alpha）
    if target in ['init_latent']:
        alpha_list = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000, 25000, 50000, 75000, 100000]
        # alpha_list = [10000, 7500,]
        use_cv = True
    elif target in ['c_clip', 'c', 'vdc_clip', 'c_vd_clip']:
        alpha_list = [100, 500, 1000, 5000, 7500, 10000, 25000, 50000, 75000, 100000, 125000, 150000]
        use_cv = True
    elif target in ['vdvae_latent', 'vdv_latent', 'v_vd_clip']:
        # alpha_list = [100, 500, 1000, 5000, 7500, 10000, 25000, 50000, 75000, 100000, 125000, 150000]
        alpha_list = [ 50000, 75000,]
        use_cv = False
    else:
        alpha_list = [75000]  # 使用固定值
        use_cv = False
    # alpha_list = [25000]  # 使用固定值
    # use_cv = False

    print(f'Using alpha list: {alpha_list}')
    
    # 3. 选择数据类型
    use_fp16 = False  # ← 是否使用 float16
    dtype = torch.float16 if use_fp16 else torch.float32
    print(f"Using dtype: {dtype}")
    
    # 4. 创建模型（指定 dtype）
    if use_cv:
        ridge = WoodburyRidgeCVTorch(
            alphas=alpha_list, 
            fit_intercept=True,
            cv=5, 
            scoring='r2', 
            device=device,
            dtype=dtype  # ← 传入数据类型
        )
    else:
        ridge = WoodburyRidgeTorch(
            alpha=alpha_list[0], 
            fit_intercept=True, 
            device=device,
            dtype=dtype  # ← 传入数据类型
        )
    
    # 5. 数据目录设置
    
    # mridir = f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/mrifeat_0526/{subject}'
    featdir = '/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/nsdfeat/subjfeat'
    # featdir = f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/extracted_features/{subject}'
    savedir = f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/decoded/congnitive/{target}/{subject}'
    os.makedirs(savedir, exist_ok=True)
    # savedir_weights = f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/decoded/ridge_weights/{subject}'
    # os.makedirs(savedir_weights, exist_ok=True)
    
    # 6. 加载数据
    print("\n========== Loading Data ==========")
    MRIDIR_MAE = f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/processed_data_hcp/{subject}'
    # train_path = f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/processed_data_hcp/{subject}/sub1_nsd_train_{ROI_NAME}_fmriavg.npy'
    # train_path = f'{MRIDIR_MAE}/nsd_train_fmriavg_{ROI_NAME}_sub{subject[-1]}.npy'
    train_path = f'{MRIDIR_MAE}/sub{subject[-1]}_nsd_train_{ROI_NAME}_fmriavg.npy'
    train_fmri = np.load(train_path)
    test_path = f'{MRIDIR_MAE}/sub{subject[-1]}_nsd_test_{ROI_NAME}_fmriavg.npy'
    # test_path = f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/processed_data_hcp/{subject}/sub1_nsd_test_{ROI_NAME}_fmriavg.npy'
    test_fmri = np.load(test_path)
    
    # Preprocessing fMRI
    train_fmri = train_fmri / 300
    test_fmri = test_fmri / 300
    
    norm_mean_train = np.mean(train_fmri, axis=0)
    norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
    X = (train_fmri - norm_mean_train) / norm_scale_train
    X_te = (test_fmri - norm_mean_train) / norm_scale_train
    X = X.astype("float32")
    X_te = X_te.astype("float32")
    
    print(f'X {X.shape}, X_te {X_te.shape}')
    print(f'X: mean={np.mean(X):.4f}, std={np.std(X):.4f}')
    print(f'X_te: mean={np.mean(X_te):.4f}, std={np.std(X_te):.4f}')
    
    # 加载目标变量
    if target == 'c_vd_vlip':
        Y = np.load(f'{featdir}/{subject}_{data_type}_vdc_clip_tr.npy').astype("float32").reshape([X.shape[0], -1])
        Y_te = np.load(f'{featdir}/{subject}_{data_type}_vdc_clip_te.npy').astype("float32").reshape([X_te.shape[0], -1])
    else:
        Y = np.load(f'{featdir}/{subject}_{data_type}_{target}_tr.npy').astype("float32").reshape([X.shape[0], -1])
        Y_te = np.load(f'{featdir}/{subject}_{data_type}_{target}_te.npy').astype("float32").reshape([X_te.shape[0], -1])
    # Y = np.load(f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/extracted_features/subj01/nsd_clipvision_train.npy').astype("float32").reshape([X.shape[0], -1])
    # Y_te = np.load(f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/extracted_features/subj01/nsd_clipvision_test.npy').astype("float32").reshape([X_te.shape[0], -1])

    print(f'\nNow making decoding model for... {subject}: {target}')
    print(f'\nProcessing roi data type: ROI: {ROI_NAME}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    print(f'Y: mean={np.mean(Y):.4f}, std={np.std(Y):.4f}')
    
    # 7. 模型训练与预测（GPU 加速）
    print("\n========== Training Woodbury Ridge (GPU) ==========")
    
    train_start = time.time()
    ridge.fit(X, Y)
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f} seconds")
    
    # 训练集预测
    train_pred = ridge.predict(X)
    if isinstance(train_pred, torch.Tensor):
        train_pred_np = train_pred.cpu().numpy()
    else:
        train_pred_np = train_pred
        
    r_train = correlation_score(Y.T, train_pred_np.T).mean()
    r_train = r_train.cpu().numpy() if hasattr(r_train, "cpu") else r_train
    print(f"Train correlation: {r_train:.4f}")
    
    # 测试集预测
    pred_start = time.time()
    scores = ridge.predict(X_te)
    pred_time = time.time() - pred_start
    print(f"Prediction time: {pred_time:.2f} seconds")
    
    if isinstance(scores, torch.Tensor):
        scores_np = scores.cpu().numpy()
    else:
        scores_np = scores
        
    r_test = correlation_score(Y_te.T, scores_np.T).mean()
    r_test = r_test.cpu().numpy() if hasattr(r_test, "cpu") else r_test
    print(f"Test correlation: {r_test:.4f}")
    
    # 计算详细评分
    rs = correlation_score(Y_te.T, scores_np.T)
    rs = rs.cpu().numpy() if hasattr(rs, "cpu") else rs
    print(f'\nPrediction accuracy: {np.mean(rs):.4f}')
    print(f'Mean r: {np.mean(rs):.4f}, Std: {np.std(rs):.4f}, Max: {np.max(rs):.4f}, Min: {np.min(rs):.4f}')
    
    # 标准化预测结果
    std_norm_score_latent = (scores_np - np.mean(scores_np, axis=0)) / np.std(scores_np, axis=0)
    pred_latents = std_norm_score_latent * np.std(Y, axis=0) + np.mean(Y, axis=0)
    
    # r2_score = ridge.score(Y_te, pred_latents)
    # print(f'R^2 score: {r2_score:.4f}')
    # ✅ 正确方案：使用 sklearn.metrics 库
    from sklearn.metrics import r2_score

    # 注意参数顺序：先传真实值，再传预测值
    # Y_te: (982, 6400)
    # pred_latents: (982, 6400)
    my_r2 = r2_score(Y_te, pred_latents)

    print(f'Corrected R^2 score: {my_r2:.4f}')

    # 额外：计算并保存权值范数
    coef_tensor = ridge.best_model_.coef_ if use_cv else ridge.coef_
    l2_norms = weight_norms(coef_tensor, "l2")
    l1_norms = weight_norms(coef_tensor, "l1")
    norms_dir = os.path.join(savedir, "weight_norms")
    norms_base = f"{subject}_{target}_{ROI_NAME}_wood"
    percentiles = save_weight_norms(norms_dir, norms_base, l1_norms, l2_norms)
    print(f"L2 norms percentiles (25/50/75): {percentiles['l2']}")
    print(f"L1 norms percentiles (25/50/75): {percentiles['l1']}")


    # 8. 保存预测结果
    np.save(f'{savedir}/{subject}_{target}_{ROI_NAME}_wood.npy', pred_latents)
    print(f"\n✓ Results saved to: {savedir}/{subject}_{target}_{ROI_NAME}_wood.npy")
    
    # 清理 GPU 内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n========== Total time: {end_time - start_time:.2f} seconds ==========")