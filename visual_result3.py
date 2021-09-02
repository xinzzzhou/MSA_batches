import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math

root_path = "/mnt/a22896c0-1177-4b70-9625-fdbdf3e9e159/PDB/data/"

def get_path(name, type, method = ""):
    if method == "":
        return f"{root_path + name[1:3]}/{name}.{type}"
    else:
        return f"{root_path + name[1:3]}/{name}_{method}.{type}"

def get_depth(all_pdbs):
    dict_tmp = dict()
    for pdb in all_pdbs:
        with open(get_path(pdb[:5],'aln')) as f:
            dict_tmp[pdb[:5]] = len(f.readlines())
    return dict_tmp

def get_length(all_pdbs):
    dict_tmp = dict()
    for pdb in all_pdbs:
        with open(get_path(pdb[:5], 'aln')) as f:
            dict_tmp[pdb[:5]] = len(f.readline())
    return dict_tmp

def get_sorted_file():
    # get depth of MSA and sort them
    file_all = "/mnt/a22896c0-1177-4b70-9625-fdbdf3e9e159/PDB/processed_len/all/"
    for _,_, fs in os.walk(file_all):
        return [tmp for tmp in fs]

def process_line(line):
    gene, acc = line.split("]\n")[0].split("\t[")
    t = []
    tmp=""
    for ac in acc:
        if ac == " ":
            if tmp != "":
                t.append(float(tmp))
                tmp = ""
        else:
                tmp += ac
    return gene, np.mean(np.array(t))


def get_dict(file):
    dict_tmp = dict()
    for line in file:
        gene, acc = process_line(line)
        dict_tmp[gene] = acc
    return dict_tmp

file_gremlin = open("/mnt/a22896c0-1177-4b70-9625-fdbdf3e9e159/PDB/msaT_result/evaluate_result/record_msaT_withpad", 'r')
# file_gremlin = open("/mnt/a22896c0-1177-4b70-9625-fdbdf3e9e159/PDB/msaT_result/evaluate_result/record_msaT_removepad", 'r')
file_msaT = open("/mnt/a22896c0-1177-4b70-9625-fdbdf3e9e159/PDB/msaT_result/evaluate_result/record_msaT_nopad",'r')
# load the data
dict_msat = get_dict(file_gremlin)
dict_gremlin = get_dict(file_msaT)
all_pdbs = get_sorted_file()
dict_length = get_length(all_pdbs)

x, y, z = [], [], []

dict_ratio = defaultdict(list)
for pdb in all_pdbs:
    acc1 = dict_msat.get(pdb[:5])
    acc2 = dict_gremlin.get(pdb[:5])
    z.append(math.log(dict_length.get(pdb[:5])))
    x.append(acc1)
    y.append(acc2)
plt.title("MSAt(withpad)_MSAt(nopad)")
# plt.title("MSAt(removepad)_MSAt(nopad)")
plt.xlabel('MSAt(withpad)')
# plt.xlabel('MSAt(removepad)')
plt.ylabel('MSAt(nopad)')
# plt.xlim(xmax=10, xmin=1)
# plt.ylim(ymax=10, ymin=1)
# a, b = list(range(np.max(np.array(z))))
# 画散点图
# cm = plt.cm.get_cmap('terrain')
cm = plt.cm.get_cmap('viridis')
# sc = plt.scatter(x, y, c=z, s=np.pi, vmin=0, vmax=20, cmap=cm)
sc = plt.scatter(x, y, c=z, s=np.pi, cmap=cm)
plt.colorbar(sc)
# plt.plot(a,b, color='black', linestyle='dashed')
plt.plot([0,np.max(np.array(y))], [0,np.max(np.array(y))], 'k--', linewidth=4)
# plt.scatter(x, y, s=area, c=colors1, alpha=0.4, label='<=10')
# plt.legend(title="MSA depth")
plt.savefig('visual_result1_withpad_nopad_length.png', dpi=300)
# plt.savefig('visual_result1_removepad_nopad_length.png', dpi=300)
plt.show()