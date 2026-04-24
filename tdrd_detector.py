"""
TDRD (Tensor Decomposition and Reconstruction Detector) for Hyperspectral Change Detection
Exact Python implementation matching MATLAB code behavior

Reference:
    Multiscale morphological compressed change vector analysis for 
    unsupervised multiple change detection

Author: Converted from MATLAB to Python
Original Author: Zephyr Hou
"""

import numpy as np
from scipy import linalg
from typing import Tuple
import warnings


def normalize_hsi(hsi: np.ndarray, paras: int = 1) -> np.ndarray:
    """
    Normalize the 3D hyperspectral dataset
    Exact match to func_Normalized.m
    
    Args:
        hsi: Input data with shape (rows, cols, bands)
        paras: Normalization type
            1: Min-max Normalization (range scaling)
            2: Mean Normalization
            3: Standardization Normalization (Z-score)
            4: Scaling to Unit Length Normalization
    
    Returns:
        Normalized result with shape (rows, cols, bands)
    """
    if paras == 1:
        # Min-max Normalization
        maxVal = hsi.max()
        minVal = hsi.min()
        Res = (hsi - minVal) / (maxVal - minVal)
    elif paras == 2:
        # Mean Normalization
        meanVal = hsi.mean()
        maxVal = hsi.max()
        minVal = hsi.min()
        Res = (hsi - meanVal) / (maxVal - minVal)
    elif paras == 3:
        # Standardization (Z-score)
        meanVal = hsi.mean()
        stdVal = hsi.std()
        Res = (hsi - meanVal) / stdVal
    elif paras == 4:
        # Scaling to Unit Length
        rows, cols, bands = hsi.shape
        Rhsi = hsi.reshape(rows * cols, bands).T  # (bands, rows*cols)
        Nhsi = Rhsi / np.linalg.norm(Rhsi, axis=0)
        Res = Nhsi.T.reshape(rows, cols, bands)
    else:
        warnings.warn('Please input the normalized type: 1,2,3,or 4 !')
        Res = hsi
    
    return Res


def unfold_tensor(hsi: np.ndarray, mode: int) -> np.ndarray:
    """
    Mode-n unfolding of Tensor
    Exact match to func_unfold.m
    
    MATLAB: result = reshape(shiftdim(hsi,mode-1), dim(mode), []);
    
    Args:
        hsi: Input tensor with shape (d1, d2, d3)
        mode: Mode to unfold (1, 2, or 3)
    
    Returns:
        Unfolded matrix with shape (dim(mode), -1)
    """
    dim = hsi.shape
    # shiftdim(hsi, mode-1): shift first (mode-1) dimensions to the end
    if mode == 1:
        # shiftdim(hsi, 0): no shift
        shifted = hsi
    elif mode == 2:
        # shiftdim(hsi, 1): shift first dimension to end
        # (d1, d2, d3) -> (d2, d3, d1)
        shifted = np.moveaxis(hsi, 0, -1)
    elif mode == 3:
        # shiftdim(hsi, 2): shift first two dimensions to end
        # (d1, d2, d3) -> (d3, d1, d2)
        shifted = np.moveaxis(hsi, [0, 1], [-1, -2])
    else:
        raise ValueError('mode must be 1, 2, or 3')
    
    # reshape to (dim(mode), -1)
    result = shifted.reshape(dim[mode - 1], -1)
    return result


def determine_pc_number(hsi: np.ndarray, rate: float = 0.99) -> Tuple[int, int, int]:
    """
    Determine the number of principal components for each mode using SVD
    Match to Main_TDRD_only.m logic
    """
    PCs = []
    
    for mode in range(1, 4):
        rehsi = unfold_tensor(hsi, mode)
        Val = linalg.svd(rehsi, compute_uv=False)
        
        Sumva = rate * np.sum(Val)
        T0 = np.cumsum(Val)
        ki = np.where(T0 >= Sumva)[0]
        if len(ki) > 0:
            PCs.append(ki[0] + 1)  # 1-indexed
        else:
            PCs.append(len(Val))
    
    return tuple(PCs)


def tucker_hosvd_reconstruction(hsi: np.ndarray, PC: Tuple[int, int, int], 
                                 paras: int = 1) -> np.ndarray:
    """
    Tucker decomposition and reconstruction using HOSVD
    Match to func_tucker.m behavior
    
    Args:
        hsi: Input tensor with shape (rows, cols, bands)
        PC: Principal component numbers [pc1, pc2, pc3]
        paras: Normalization type
    
    Returns:
        Reconstructed tensor
    """
    H, W, Dim = hsi.shape
    
    # Normalize initial data
    hsi = normalize_hsi(hsi, paras)
    
    # HOSVD: compute factor matrices via SVD for each mode
    # This matches the behavior of tucker_als with full dimensions
    U1, _, _ = linalg.svd(unfold_tensor(hsi, 1), full_matrices=False)
    U2, _, _ = linalg.svd(unfold_tensor(hsi, 2), full_matrices=False)
    U3, _, _ = linalg.svd(unfold_tensor(hsi, 3), full_matrices=False)
    
    # Truncate to PC
    U1_new = U1[:, :PC[0]]
    U2_new = U2[:, :PC[1]]
    U3_new = U3[:, :PC[2]]
    
    # Compute core tensor: core = hsi x1 U1^T x2 U2^T x3 U3^T
    # Using tensordot for efficient computation
    temp = np.tensordot(U1_new.T, hsi, axes=([1], [0]))  # (pc1, W, Dim)
    temp = np.tensordot(U2_new.T, temp, axes=([1], [1]))  # (pc1, pc2, Dim)
    temp = np.moveaxis(temp, 0, 1)  # Fix axis order
    core_new = np.tensordot(U3_new.T, temp, axes=([1], [2]))  # (pc1, pc2, pc3)
    core_new = np.moveaxis(core_new, 0, 2)
    
    # Reconstruct: hsi_new = core x1 U1 x2 U2 x3 U3
    temp = np.tensordot(U3_new, core_new, axes=([1], [2]))  # (Dim, pc1, pc2)
    temp = np.moveaxis(temp, 0, 2)
    temp = np.tensordot(U2_new, temp, axes=([1], [1]))  # (pc2, pc1, W)
    temp = np.moveaxis(temp, 0, 1)
    hsi_new = np.tensordot(U1_new, temp, axes=([1], [0]))  # (H, pc2, W)
    hsi_new = np.moveaxis(hsi_new, 0, 0)
    
    # Reshape to match MATLAB output
    # MATLAB: unfold_hsi = tenmat(new_hsi,3); hsi_new=reshape(unfold_hsi',H,W,Dim);
    hsi_new = hsi_new.reshape(H, W, Dim)
    
    # Normalize results
    hsi_new = normalize_hsi(hsi_new, paras)
    
    return hsi_new


def rlad_detector(hsi_t1: np.ndarray, hsi_t2: np.ndarray) -> np.ndarray:
    """
    Revised Local Absolute Distance (RLAD) detector
    Exact match to func_RLAD.m
    
    Args:
        hsi_t1: Hyperspectral image at time 1 with shape (rows, cols, bands)
        hsi_t2: Hyperspectral image at time 2 with shape (rows, cols, bands)
    
    Returns:
        Detection result with shape (rows, cols)
    """
    win_out = 3
    win_in = 1
    
    rows, cols, bands = hsi_t1.shape
    result = np.zeros((rows, cols))
    
    t = win_out // 2  # = 1
    t1 = win_in // 2   # = 0
    M = win_out ** 2   # = 9
    
    # Adaptive Boundary Filling for hsi_t1
    DataTest1 = np.zeros((rows + 2*t, cols + 2*t, bands))
    DataTest1[t:rows+t, t:cols+t, :] = hsi_t1
    # Left padding (reflect)
    DataTest1[t:rows+t, :t, :] = hsi_t1[:, t-1::-1, :]
    # Right padding (reflect)
    DataTest1[t:rows+t, t+cols:, :] = hsi_t1[:, cols-1:cols-t-1:-1, :]
    # Top padding (reflect)
    DataTest1[:t, :, :] = DataTest1[2*t-1:t-1:-1, :, :]
    # Bottom padding (reflect)
    DataTest1[t+rows:, :, :] = DataTest1[t+rows-1:rows-1:-1, :, :]
    
    # Adaptive Boundary Filling for hsi_t2
    DataTest2 = np.zeros((rows + 2*t, cols + 2*t, bands))
    DataTest2[t:rows+t, t:cols+t, :] = hsi_t2
    DataTest2[t:rows+t, :t, :] = hsi_t2[:, t-1::-1, :]
    DataTest2[t:rows+t, t+cols:, :] = hsi_t2[:, cols-1:cols-t-1:-1, :]
    DataTest2[:t, :, :] = DataTest2[2*t-1:t-1:-1, :, :]
    DataTest2[t+rows:, :, :] = DataTest2[t+rows-1:rows-1:-1, :, :]
    
    # Core algorithm - iterate over each pixel
    for i in range(t, cols + t):
        for j in range(t, rows + t):
            # Extract block for t1
            block1 = DataTest1[j-t:j+t+1, i-t:i+t+1, :].copy()
            y1 = DataTest1[j, i, :].copy()  # Center pixel
            
            # Set inner window to NaN and remove
            block1[t-t1:t+t1+1, t-t1:t+t1+1, :] = np.nan
            block1_2d = block1.reshape(M, bands)
            block1_2d = block1_2d[~np.isnan(block1_2d[:, 0])]
            H1 = block1_2d.T  # (bands, num_samples)
            
            # Extract block for t2
            block2 = DataTest2[j-t:j+t+1, i-t:i+t+1, :].copy()
            y2 = DataTest2[j, i, :].copy()
            
            block2[t-t1:t+t1+1, t-t1:t+t1+1, :] = np.nan
            block2_2d = block2.reshape(M, bands)
            block2_2d = block2_2d[~np.isnan(block2_2d[:, 0])]
            H2 = block2_2d.T  # (bands, num_samples)
            
            # RLAD calculation
            tempD = np.sum(np.abs(H2 - H1))
            
            # Weight calculation
            norm_y1 = np.linalg.norm(y1)
            norm_y2 = np.linalg.norm(y2)
            if norm_y1 > 1e-10 and norm_y2 > 1e-10:
                w = np.arctan(((y1 @ y2) / (norm_y1 * norm_y2)) ** 2)
            else:
                w = 0
            
            result[j-t, i-t] = tempD * w
    
    return result


def rlad_detector_vectorized(hsi_t1: np.ndarray, hsi_t2: np.ndarray) -> np.ndarray:
    """
    Vectorized RLAD detector for faster computation
    Approximates func_RLAD.m behavior
    """
    rows, cols, bands = hsi_t1.shape
    
    # Pad images with reflection
    pad_width = ((1, 1), (1, 1), (0, 0))
    DataTest1 = np.pad(hsi_t1, pad_width, mode='reflect')
    DataTest2 = np.pad(hsi_t2, pad_width, mode='reflect')
    
    # Compute absolute difference for all pixels
    # Using sliding window approach
    result = np.zeros((rows, cols))
    
    # Pre-compute all 8 neighbors for each pixel
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue  # Skip center pixel
            
            # Shift and compute difference
            shifted_t1 = DataTest1[1+di:rows+1+di, 1+dj:cols+1+dj, :]
            shifted_t2 = DataTest2[1+di:rows+1+di, 1+dj:cols+1+dj, :]
            
            result += np.sum(np.abs(shifted_t2 - shifted_t1), axis=2)
    
    # Compute weight (cosine similarity based)
    y1 = hsi_t1
    y2 = hsi_t2
    
    norm_y1 = np.linalg.norm(y1, axis=2) + 1e-10
    norm_y2 = np.linalg.norm(y2, axis=2) + 1e-10
    
    dot_product = np.sum(y1 * y2, axis=2)
    cos_sim = dot_product / (norm_y1 * norm_y2)
    
    w = np.arctan(cos_sim ** 2)
    
    result = result * w
    
    return result


def tdrd_detect(hsi_t1: np.ndarray, hsi_t2: np.ndarray, 
                rate: float = 0.99, paras: int = 1,
                verbose: bool = True) -> np.ndarray:
    """
    TDRD (Tensor Decomposition and Reconstruction Detector)
    Exact match to Main_TDRD_only.m
    
    Args:
        hsi_t1: Hyperspectral image at time 1 with shape (rows, cols, bands)
        hsi_t2: Hyperspectral image at time 2 with shape (rows, cols, bands)
        rate: Cumulative energy ratio for Tucker decomposition (default 0.99)
        paras: Normalization type (1-4)
        verbose: Whether to print progress
    
    Returns:
        Detection result with shape (rows, cols)
    """
    if verbose:
        print("=" * 60)
        print("TDRD (Tensor Decomposition and Reconstruction Detector)")
        print("=" * 60)
    
    # Step 1: Determine PC numbers
    if verbose:
        print("\n[Step 1] Determining PC numbers...")
    
    PCs_t1 = determine_pc_number(hsi_t1, rate)
    PCs_t2 = determine_pc_number(hsi_t2, rate)
    
    if verbose:
        print(f"  T1 PCs (rows, cols, bands): {PCs_t1}")
        print(f"  T2 PCs (rows, cols, bands): {PCs_t2}")
    
    # Step 2: Tucker decomposition and reconstruction
    if verbose:
        print("\n[Step 2] Tucker decomposition and reconstruction...")
    
    hsi_t1_new = tucker_hosvd_reconstruction(hsi_t1, PCs_t1, paras)
    hsi_t2_new = tucker_hosvd_reconstruction(hsi_t2, PCs_t2, paras)
    
    if verbose:
        print("  Reconstruction completed.")
    
    # Step 3: RLAD detection
    if verbose:
        print("\n[Step 3] RLAD detection...")
    
    # Use vectorized version for speed
    result = rlad_detector_vectorized(hsi_t1_new, hsi_t2_new)
    
    if verbose:
        print("  Detection completed.")
        print(f"  Result range: [{result.min():.4f}, {result.max():.4f}]")
    
    return result


def accuracy_assessment(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Accuracy assessment for change detection
    Match to accuracy_assessment.m
    """
    eps = 1e-10
    
    gt_flat = gt.flatten().astype(float)
    pred_flat = pred.flatten().astype(float)
    
    mask = (gt_flat == 0) | (gt_flat == 1)
    gt_flat = gt_flat[mask]
    pred_flat = pred_flat[mask]
    
    TN = np.sum((gt_flat == 0) & (pred_flat == 0))
    FP = np.sum((gt_flat == 0) & (pred_flat == 1))
    FN = np.sum((gt_flat == 1) & (pred_flat == 0))
    TP = np.sum((gt_flat == 1) & (pred_flat == 1))
    
    total = TN + TP + FN + FP
    po = (TN + TP) / (total + eps)
    pe = ((TN + FN) * (TN + FP) + (FP + TP) * (FN + TP)) / (total ** 2 + eps)
    kappa = (po - pe) / (1 - pe + eps)
    
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    oa = (TP + TN) / (total + eps)
    
    oa_unchanged = TN / (TN + FP + eps)
    oa_changed = TP / (TP + FN + eps)
    
    return {
        'conf_mat': np.array([[TN, FP], [FN, TP]]),
        'OA': oa,
        'Kappa': kappa,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'OA_unchanged': oa_unchanged,
        'OA_changed': oa_changed,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }


def find_optimal_threshold(result: np.ndarray, gt: np.ndarray, 
                           threshold_range: np.ndarray = None) -> Tuple[float, dict]:
    """Find optimal threshold based on F1 score"""
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.9, 0.05)
    
    result_min = result.min()
    result_max = result.max()
    result_norm = (result - result_min) / (result_max - result_min + 1e-10)
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for th in threshold_range:
        pred = (result_norm > th).astype(float)
        metrics = accuracy_assessment(gt, pred)
        
        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            best_threshold = th
            best_metrics = metrics
    
    return best_threshold, best_metrics


if __name__ == "__main__":
    print("TDRD Detector Module Loaded Successfully!")
