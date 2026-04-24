"""
Python Implementation of the TDRD Algorithm
"""

import os
import sys
import time
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdrd_detector import tdrd_detect, find_optimal_threshold, accuracy_assessment


def Main_TDRD():
    """Test TDRD on Farmland dataset"""
    print("=" * 70)
    print("TDRD Test")
    print("=" * 70)
    
    # Load data
    data_path = r'C:\Users\zephy\Documents\Datasets\CDdataset_Farmland_450x140x155.mat'
    data = loadmat(data_path)
    
    hsi_t1 = data['hsi_t1'].astype(np.float64)
    hsi_t2 = data['hsi_t2'].astype(np.float64)
    gt = data['hsi_gt'].astype(np.float64)
    
    print(f"\nDataset shape: {hsi_t1.shape}")
    print(f"GT changed ratio: {gt.mean()*100:.2f}%")
    
    # Run TDRD
    start_time = time.time()
    result = tdrd_detect(hsi_t1, hsi_t2, rate=0.99, paras=1, verbose=True)
    elapsed_time = time.time() - start_time
    
    print(f"\nTotal time: {elapsed_time:.2f}s")
    
    # Find optimal threshold
    best_threshold, best_metrics = find_optimal_threshold(result, gt)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"OA: {best_metrics['OA']*100:.2f}% ")
    print(f"Kappa: {best_metrics['Kappa']:.4f} ")
    print(f"F1: {best_metrics['F1']*100:.2f}% ")
    print(f"Precision: {best_metrics['Precision']*100:.2f}% ")
    print(f"Recall: {best_metrics['Recall']*100:.2f}% ")
    print(f"OA (Changed): {best_metrics['OA_changed']*100:.2f}%")
    print(f"OA (Unchanged): {best_metrics['OA_unchanged']*100:.2f}%")
    

if __name__ == "__main__":
    Main_TDRD()
