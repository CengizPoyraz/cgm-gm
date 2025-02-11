"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from gms.gms_utils import get_gms_path
  
def visualization (ori_data, generated_data, analysis, args, run=None):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]
  

  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  print(f'[raw] ori_data check -> shape={ori_data.shape} NaN exists: {np.any(np.isnan(ori_data))} Nan count: {np.count_nonzero(np.isnan(ori_data))}')
  print(f'[raw] ori_data sample={generated_data[0]}')
  print(f'[raw] generated_data check -> shape={generated_data.shape} NaN exists: {np.any(np.isnan(generated_data))} Nan count: {np.count_nonzero(np.isnan(generated_data))}')
  print(f'[raw] generated_data sample={generated_data[0]}')

  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  
  print(f'[idx] ori_data check -> shape={ori_data.shape} NaN exists: {np.any(np.isnan(ori_data))} Nan count: {np.count_nonzero(np.isnan(ori_data))}')
  print(f'[idx] ori_data sample={ori_data[0]}')
  print(f'[idx] generated_data check -> shape={generated_data.shape} NaN exists: {np.any(np.isnan(generated_data))} Nan count: {np.count_nonzero(np.isnan(generated_data))}')
  print(f'[idx] generated_data sample={generated_data[0]}')
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
  # Visualization parameter        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()  
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.savefig(get_gms_path(f'figures/sine_pca_{args.loc}_{args.tcondvar}.png'), transparent=True,
                bbox_inches='tight', pad_inches=0, dpi=300)
    run["test/pca_ori_gen_figure"].upload(get_gms_path(f'figures/sine_pca_{args.loc}_{args.tcondvar}.png'))
    if run:
      run["test/pca_ori_gen"].log(f)
    plt.show()

    
  elif analysis == 'tsne':
    
    print(f'[tsne] prep_data check -> shape={prep_data.shape} NaN exists: {np.any(np.isnan(prep_data))} Nan count: {np.count_nonzero(np.isnan(prep_data))}')
    print(f'[tsne] prep_data_hat check -> shape={prep_data_hat.shape} NaN exists: {np.any(np.isnan(prep_data_hat))} Nan count: {np.count_nonzero(np.isnan(prep_data_hat))}')

    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    print(f'[tsne] prep_data_final check -> shape={prep_data_final.shape} NaN exists: {np.any(np.isnan(prep_data_final))} Nan count: {np.count_nonzero(np.isnan(prep_data_final))}')

    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig(get_gms_path(f'figures/sine_tsne_{args.loc}_{args.tcondvar}.png'), transparent=True,
                bbox_inches='tight', pad_inches=0, dpi=300)
    run["test/tsne_ori_gen_figure"].upload(get_gms_path(f'figures/sine_tsne_{args.loc}_{args.tcondvar}.png'))
    if run:
      run["test/tsne_ori_gen"].log(f)
    plt.show()