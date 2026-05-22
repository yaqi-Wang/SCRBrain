import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

'''
对齐sdvae和fMRI特征的模型
- fMRI 分支：MLP（到 4×40×40）
- SDVAE 分支：MLP→CNN（上采样到 4×40×40）
- 解码器：MLP
新增：fMRI 预处理（/300 + 训练集 z-score，ddof=1）
'''

# --- 1. 配置参数 ---
subject = 'subj07'
ROI_NAME = 's0_vis'  # lowvis visual other all

# 数据路径
FEATDIR = '/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/nsdfeat/subjfeat'
MRIDIR_MAE = f'/home/data/wangyaqi/projects/05Brain_Diffusser_New/01brain-diffuser/data/processed_data_hcp_mask/{subject}'
# 输出文件
OUTPUT_DIR = f'/home/data/wangyaqi/projects/10StableDiffusionReconstruction/10StableDiffusionReconstruction-main/decoded/congnitive/inforce_latent/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# fMRI输入文件
# FMRI_TRAIN_PATH = f'{MRIDIR_MAE}/sub{subject[-1]}_nsd_train_{ROI_NAME}_fmriavg.npy'
# FMRI_TEST_PATH  = f'{MRIDIR_MAE}/sub{subject[-1]}_nsd_test_{ROI_NAME}_fmriavg.npy'
FMRI_TRAIN_PATH = f'{MRIDIR_MAE}/nsd_train_fmriavg_{ROI_NAME}_sub{subject[-1]}.npy'
FMRI_TEST_PATH  = f'{MRIDIR_MAE}/nsd_test_fmriavg_{ROI_NAME}_sub{subject[-1]}.npy'
# SDVAE输入文件
SDVAE_IMAGE_TRAIN_PATH = f'{FEATDIR}/{subject}_ave_init_latent_tr.npy'
SDVAE_IMAGE_TEST_PATH  = f'{FEATDIR}/{subject}_ave_init_latent_te.npy'

# --- 共同目标表征参数 ---
COMMON_TARGET_C = 4
COMMON_TARGET_H = 40
COMMON_TARGET_W = 40
COMMON_TARGET_FLAT_DIM = COMMON_TARGET_C * COMMON_TARGET_H * COMMON_TARGET_W  # 6400

# InfoNCE和重建损失相关超参数
D_INFONCE = 1024
TEMPERATURE = 0.07

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH_SDVAE_FMRI = "sdvae_fmri_alignment_model.pth"

WEIGHT_ALIGN = 1  # 对齐损失的权重
WEIGHT_RECON = 5  # SDVAE重建损失的权重

# --- 2. 数据加载与预处理 ---
def load_and_preprocess_data(path, description=""):
    """加载.npy并展平为2D，同时返回原始形状。"""
    if not os.path.exists(path):
        print(f"错误: {description} 文件未找到 {path}")
        return None, None
    data = np.load(path)
    original_shape_info = data.shape
    if data.ndim > 2:
        data_flat = data.reshape(data.shape[0], -1).astype(np.float32)
    elif data.ndim == 2:
        data_flat = data.astype(np.float32)
    else:
        print(f"警告: {description} 数据维度为 {data.ndim}，将尝试处理。路径: {path}")
        if data.ndim == 1:
            data_flat = data.reshape(-1, 1).astype(np.float32)
        else:
            print(f"错误: {description} 数据维度无法处理。路径: {path}")
            return None, None
    print(f"成功加载 {description} 数据: {path}, 原始形状: {original_shape_info}, 展平后形状: {data_flat.shape}")
    return data_flat, original_shape_info

def load_and_preprocess_fmri(train_path, test_path, eps=1e-6):
    """专用于 fMRI 的预处理：/300 后按训练集统计量做 z-score（ddof=1）"""
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print(f"错误: fMRI 文件未找到: {train_path} 或 {test_path}")
        return None, None, None, None
    train = np.load(train_path).astype(np.float32)
    test  = np.load(test_path).astype(np.float32)

    # 全局缩放
    train = train / 300.0
    test  = test  / 300.0

    # 按特征 z-score：使用训练集的 mean/std
    norm_mean_train = np.mean(train, axis=0)
    norm_scale_train = np.std(train, axis=0, ddof=1)
    norm_scale_train = np.where(norm_scale_train < eps, 1.0, norm_scale_train)  # 防止除零

    X    = ((train - norm_mean_train) / norm_scale_train).astype(np.float32)
    X_te = ((test  - norm_mean_train) / norm_scale_train).astype(np.float32)

    print(f'预处理后的 fMRI: X {X.shape}, X_te {X_te.shape}')
    print(f'X: mean={np.mean(X):.4f}, std={np.std(X):.4f}')
    print(f'X_te: mean={np.mean(X_te):.4f}, std={np.std(X_te):.4f}')

    # 保存训练统计量，便于复现
    # np.save(os.path.join(OUTPUT_DIR, f'{subject}_{ROI_NAME}_fmri_norm_mean.npy'), norm_mean_train)
    # np.save(os.path.join(OUTPUT_DIR, f'{subject}_{ROI_NAME}_fmri_norm_std.npy'),  norm_scale_train)
    return X, X_te, norm_mean_train, norm_scale_train

class PairedSDVAEFMRIDataset(Dataset):
    """成对的 SDVAE / fMRI 特征"""
    def __init__(self, sdvae_data_flat, fmri_data_flat):
        assert sdvae_data_flat.shape[0] == fmri_data_flat.shape[0], "SDVAE和fMRI数据集的样本数量必须一致"
        self.sdvae_data = torch.from_numpy(sdvae_data_flat)
        self.fmri_data = torch.from_numpy(fmri_data_flat)
    def __len__(self):
        return self.sdvae_data.shape[0]
    def __getitem__(self, idx):
        return self.sdvae_data[idx], self.fmri_data[idx]

# --- 3. 损失函数定义 ---
def info_nce_loss(features1, features2, temperature=0.1):
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    similarity_matrix = torch.matmul(features1, features2.T) / temperature
    labels = torch.arange(features1.shape[0], device=features1.device)
    loss_i = F.cross_entropy(similarity_matrix, labels)
    loss_j = F.cross_entropy(similarity_matrix.T, labels)
    return (loss_i + loss_j) / 2.0

# --- 3. 模型定义 ---
class SDVAE_FMRI_AlignmentModel(nn.Module):
    def __init__(self, sdvae_input_flat_dim, fmri_input_flat_dim):
        super().__init__()
        self.sdvae_input_flat_dim = sdvae_input_flat_dim
        self.fmri_input_flat_dim = fmri_input_flat_dim

        # fMRI编码器: MLP -> 共享空间 (4*40*40)
        self.fmri_encoder_mlp = nn.Sequential(
            nn.Linear(fmri_input_flat_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, COMMON_TARGET_FLAT_DIM)
        )

        # SDVAE编码器: MLP -> reshape -> CNN -> (4,40,40)
        SDVAE_CNN_INTERMEDIATE_CHANNELS = 16
        SDVAE_CNN_INTERMEDIATE_H = 20
        SDVAE_CNN_INTERMEDIATE_W = 20
        SDVAE_MLP_OUTPUT_DIM = SDVAE_CNN_INTERMEDIATE_CHANNELS * SDVAE_CNN_INTERMEDIATE_H * SDVAE_CNN_INTERMEDIATE_W

        self.sdvae_encoder_mlp = nn.Sequential(
            nn.Linear(sdvae_input_flat_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, SDVAE_MLP_OUTPUT_DIM)
        )
        self.sdvae_encoder_cnn = nn.Sequential(
            nn.Conv2d(SDVAE_CNN_INTERMEDIATE_CHANNELS, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, COMMON_TARGET_C, kernel_size=1, stride=1, padding=0)
        )
        self.sdvae_cnn_intermediate_channels = SDVAE_CNN_INTERMEDIATE_CHANNELS
        self.sdvae_cnn_intermediate_h = SDVAE_CNN_INTERMEDIATE_H
        self.sdvae_cnn_intermediate_w = SDVAE_CNN_INTERMEDIATE_W

        # InfoNCE 投影头
        proj_head_intermediate_dim = COMMON_TARGET_FLAT_DIM  # 6400
        self.sdvae_proj_head = nn.Sequential(
            nn.Linear(COMMON_TARGET_FLAT_DIM, proj_head_intermediate_dim), nn.ReLU(),
            nn.Linear(proj_head_intermediate_dim, D_INFONCE)
        )
        self.fmri_proj_head = nn.Sequential(
            nn.Linear(COMMON_TARGET_FLAT_DIM, proj_head_intermediate_dim), nn.ReLU(),
            nn.Linear(proj_head_intermediate_dim, D_INFONCE)
        )

        # 解码器：MLP（共享表征 -> SDVAE原始维度）
        self.sdvae_decoder_mlp = nn.Sequential(
            nn.Linear(COMMON_TARGET_FLAT_DIM, 2048),
            nn.ReLU(),
            nn.Linear(2048, sdvae_input_flat_dim)
        )

    def forward(self, sdvae_flat, fmri_flat):
        # fMRI: MLP -> (N,6400) -> (N,4,40,40)
        fmri_encoded_flat = self.fmri_encoder_mlp(fmri_flat)
        fmri_encoded_target_shape = fmri_encoded_flat.view(-1, COMMON_TARGET_C, COMMON_TARGET_H, COMMON_TARGET_W)

        # SDVAE: MLP -> (N, Cmid*20*20) -> CNN -> (N,4,40,40)
        sdvae_mlp_out_flat = self.sdvae_encoder_mlp(sdvae_flat)
        sdvae_cnn_input_shape = sdvae_mlp_out_flat.view(
            -1, self.sdvae_cnn_intermediate_channels,
            self.sdvae_cnn_intermediate_h, self.sdvae_cnn_intermediate_w
        )
        sdvae_encoded_target_shape = self.sdvae_encoder_cnn(sdvae_cnn_input_shape)

        # InfoNCE 投影
        fmri_encoded_flat_for_proj = fmri_encoded_target_shape.view(fmri_encoded_target_shape.size(0), -1)
        sdvae_encoded_flat_for_proj = sdvae_encoded_target_shape.view(sdvae_encoded_target_shape.size(0), -1)
        projected_fmri_for_infonce = self.fmri_proj_head(fmri_encoded_flat_for_proj)
        projected_sdvae_for_infonce = self.sdvae_proj_head(sdvae_encoded_flat_for_proj)

        # 融合（fMRI 主导）并解码回 SDVAE 空间
        fmri_sdvae_encoded_target_shape = 0.01 * fmri_encoded_target_shape + 1 * sdvae_encoded_target_shape
        fmri_encoded_target_flat = fmri_sdvae_encoded_target_shape.view(-1, COMMON_TARGET_FLAT_DIM)
        reconstructed_sdvae_flat = self.sdvae_decoder_mlp(fmri_encoded_target_flat)

        return projected_sdvae_for_infonce, projected_fmri_for_infonce, reconstructed_sdvae_flat

# --- 4. 主逻辑 ---
def main():
    print(f"使用的设备: {DEVICE}")

    # --- 4.1 加载训练数据 ---
    print("\n--- 正在加载训练数据 ---")
    # SDVAE（保持原函数）
    sdvae_train_flat, sdvae_train_original_shape = load_and_preprocess_data(SDVAE_IMAGE_TRAIN_PATH, "SDVAE训练集")

    # fMRI（使用专用预处理：/300 + 训练集统计量 z-score）
    fmri_train_flat, fmri_test_flat, norm_mean_train, norm_scale_train = load_and_preprocess_fmri(
        FMRI_TRAIN_PATH, FMRI_TEST_PATH
    )

    if sdvae_train_flat is None or fmri_train_flat is None:
        print("错误：一个或多个训练数据文件加载失败。程序将退出。")
        return

    sdvae_input_flat_dim = sdvae_train_flat.shape[1]
    fmri_input_flat_dim = fmri_train_flat.shape[1]

    train_dataset = PairedSDVAEFMRIDataset(sdvae_train_flat, fmri_train_flat)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=min(4, os.cpu_count() or 1), pin_memory=True)

    # --- 4.2 初始化和训练模型 ---
    model = SDVAE_FMRI_AlignmentModel(sdvae_input_flat_dim, fmri_input_flat_dim).to(DEVICE)
    print("\n模型结构:")
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("\n--- 开始训练模型 ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_epoch_loss = 0.0
        total_epoch_loss_align = 0.0
        total_epoch_loss_recon = 0.0

        for sdvae_batch_flat, fmri_batch_flat in train_loader:
            sdvae_batch_flat = sdvae_batch_flat.to(DEVICE)  # 原始 SDVAE 输入
            fmri_batch_flat = fmri_batch_flat.to(DEVICE)

            optimizer.zero_grad()
            projected_sdvae_for_infonce, projected_fmri_for_infonce, reconstructed_sdvae_flat = model(
                sdvae_batch_flat, fmri_batch_flat
            )

            recon_loss_val = F.mse_loss(reconstructed_sdvae_flat, sdvae_batch_flat)
            align_loss_val = info_nce_loss(projected_sdvae_for_infonce, projected_fmri_for_infonce, temperature=TEMPERATURE)

            combined_loss = WEIGHT_ALIGN * align_loss_val + WEIGHT_RECON * recon_loss_val
            combined_loss.backward()
            optimizer.step()

            total_epoch_loss += combined_loss.item()
            total_epoch_loss_align += align_loss_val.item()
            total_epoch_loss_recon += recon_loss_val.item()

        avg_epoch_loss = total_epoch_loss / len(train_loader)
        avg_epoch_loss_align = total_epoch_loss_align / len(train_loader)
        avg_epoch_loss_recon = total_epoch_loss_recon / len(train_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Total Loss: {avg_epoch_loss:.6f}, "
              f"Avg Align (InfoNCE): {avg_epoch_loss_align:.6f}, Avg Recon (MSE): {avg_epoch_loss_recon:.6f}")

    print("\n--- 训练完成! ---")
    torch.save(model.state_dict(), SAVE_MODEL_PATH_SDVAE_FMRI)
    print(f"模型已保存到: {SAVE_MODEL_PATH_SDVAE_FMRI}")

    # --- 4.3 加载并处理测试集数据 ---
    print("\n--- 开始处理测试集数据 ---")
    # SDVAE 测试集
    sdvae_test_flat, sdvae_test_original_shape = load_and_preprocess_data(SDVAE_IMAGE_TEST_PATH, "SDVAE测试集")
    # fMRI 测试集（已在上面一次性预处理得到 fmri_test_flat）
    if sdvae_test_flat is None or fmri_test_flat is None:
        print("警告：一个或多个测试数据文件加载失败，将跳过测试集处理。")
    else:
        model.eval()
        with torch.no_grad():
            sdvae_test_tensor_flat = torch.from_numpy(sdvae_test_flat).to(DEVICE)
            fmri_test_tensor_flat = torch.from_numpy(fmri_test_flat).to(DEVICE)

            projected_sdvae_for_test_infonce, projected_fmri_for_test_infonce, reconstructed_sdvae_test_flat = model(
                sdvae_test_tensor_flat, fmri_test_tensor_flat
            )
            test_loss_infonce = info_nce_loss(projected_sdvae_for_test_infonce, projected_fmri_for_test_infonce, temperature=TEMPERATURE)
            test_loss_recon = F.mse_loss(reconstructed_sdvae_test_flat, sdvae_test_tensor_flat)
            test_combined_loss = WEIGHT_ALIGN * test_loss_infonce + WEIGHT_RECON * test_loss_recon

            reconstructed_sdvae_test_numpy_flat = reconstructed_sdvae_test_flat.cpu().numpy()

            print(f"\n测试集评估结果:")
            print(f"  测试集InfoNCE损失: {test_loss_infonce.item():.6f}")
            print(f"  测试集重建MSE损失: {test_loss_recon.item():.6f}")
            print(f"  测试集加权总损失: {test_combined_loss.item():.6f}")

        print(f"\n原始测试集SDVAE (展平后) 形状: {sdvae_test_flat.shape}")
        print(f"由fMRI重建的测试集SDVAE (展平后) 形状: {reconstructed_sdvae_test_numpy_flat.shape}")

        assert reconstructed_sdvae_test_numpy_flat.shape == sdvae_test_flat.shape, \
            "重建的测试集SDVAE特征形状与原始展平后的测试集SDVAE特征形状不一致!"
        print("重建的测试集SDVAE特征形状与原始展平后的SDVAE特征形状一致。")

        output_filename_flat     = f'{OUTPUT_DIR}/{subject}_inforce_latent_{ROI_NAME}_wood_flat.npy'
        output_filename_reshaped = f'{OUTPUT_DIR}/{subject}_inforce_latent_{ROI_NAME}_wood_reshaped.npy'

        # 总是保存展平版
        np.save(output_filename_flat, reconstructed_sdvae_test_numpy_flat)
        print(f"由fMRI重建的测试集SDVAE (展平) 已保存到: {output_filename_flat}")

        # 若原始是多维且维度匹配，再额外保存“恢复形状版”
        if sdvae_test_original_shape is not None and len(sdvae_test_original_shape) > 2:
            num_samples_test = sdvae_test_original_shape[0]
            original_dims_sdvae_test = sdvae_test_original_shape[1:]
            expected_flat_dim_test = int(np.prod(original_dims_sdvae_test))
            if reconstructed_sdvae_test_numpy_flat.shape[1] == expected_flat_dim_test:
                reconstructed_sdvae_test_output_reshaped = reconstructed_sdvae_test_numpy_flat.reshape(
                    num_samples_test, *original_dims_sdvae_test
                )
                print(f"由fMRI重建的测试集SDVAE (恢复原始形状后) 形状: {reconstructed_sdvae_test_output_reshaped.shape}")
                np.save(output_filename_reshaped, reconstructed_sdvae_test_output_reshaped)
                print(f"由fMRI重建的测试集SDVAE (恢复形状) 已保存到: {output_filename_reshaped}")
            else:
                print(f"注意: 无法将重建的测试集SDVAE恢复到原始形状 {sdvae_test_original_shape}，"
                      f"展平维度不匹配 ({reconstructed_sdvae_test_numpy_flat.shape[1]} != {expected_flat_dim_test})。")

    print("\n--- 所有处理完成 ---")

if __name__ == "__main__":
    main()
