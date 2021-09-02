import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_acc(pred, true, sub_ln):
    pred_true = np.stack((pred,true),-1)
    pred_true = pred_true[pred_true[:, 0].argsort()[::-1]]
    acc = []
    for n in range(10):
      N = int(sub_ln * (n+1)/10)
      tmp = pred_true[:N, 1]
      acc.append(np.mean(tmp))
    return np.array(acc)

# msa_name = []
# with open('D:\msa_dataset\list', 'r', encoding='utf-8') as f:
#     for line in f:
#         line = line.split(' ')
#         msa_name.append(line[0])
#     f.close()

with open('/usr/data/txc/esm-master/examples/data/data/A6/1A62A.mtx', 'r', encoding='utf-8') as pred:
    gremlin_pred = np.loadtxt(pred)
    pred.close()
# plt.imshow(gremlin_pred)
# plt.colorbar()
# plt.show()
cst_file = open('/usr/data/txc/esm-master/examples/data/data/A6/1A62A.dist', 'r', encoding='utf-8')
cst_pd = np.array(pd.read_csv(cst_file, sep=' ', header=None, usecols=[0, 2, 4], encoding='utf-8'))
ref_file = open('/usr/data/txc/esm-master/examples/data/data/A6/1A62A.mtx_ref', 'r', encoding='utf-8').readlines()
ref_seq_ = ref_file[1].strip()
# calculate the length of sequence been used
gap_dict = {}
ncol = 0
for idx, AA in enumerate(ref_seq_):
    if AA != "-":
        gap_dict[idx] = ncol
        ncol = ncol + 1
cst_np = list()
for line in cst_pd:
    index_i = int(line[0])
    index_j = int(line[1])
    AA_i = ref_seq_[index_i]
    AA_j = ref_seq_[index_j]
    if AA_i != "-" and AA_j != "-" and index_i >= 0 and index_j >= 0:
        cst_np.append([gap_dict[index_i], gap_dict[index_j], line[2]])
cst_true = np.array(cst_np)
cst_mtx = np.ones((ncol, ncol)) * 10000
# cst_mtx = np.zeros((ncol, ncol))
for a, pair in enumerate(cst_true[:, 0:2].astype(int)):
    cst_mtx[pair[0], pair[1]] = cst_true[a, 2]
true = (cst_mtx < 5).astype(int)
# plt.imshow(true + true.T)
# plt.colorbar()
# plt.show()

idx = np.triu_indices_from(true, k=6)
acc = get_acc(gremlin_pred[idx], true[idx], ncol)

plt.plot(acc)
plt.show()
