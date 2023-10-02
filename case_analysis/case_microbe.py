from scipy.sparse import coo_matrix
import numpy as np
import torch
import os

# import numpy as np
# top candidate ls
def topk_candidate_microbles2file(topk= 20, epoch= 79):

	microbe_names= []
	with open('../data/microbe_names.txt', 'r', encoding= 'utf-8') as f:
		for line in f.readlines():
			microbe_names.append(line.rstrip())
	microbe_names= np.array(microbe_names)
	drug_names= []
	with open('../data/drug_names.txt', 'r', encoding= 'utf-8') as f:
		for line in f.readlines():
			drug_names.append(line.rstrip().strip('\ufeff'))
	drug_names= np.array(drug_names)
	microbe_result= []
	microbe_result= np.loadtxt(f'./run_epoch/{epoch}/case_analysis_result_all_drugs.txt')
	microbe_idx= np.argsort(-microbe_result, axis= 1)[:, 0: topk]
	ls= []
	for i in range(len(drug_names)):
		di_names= np.array([drug_names[i]]* topk).reshape(-1, 1)
		di_microbe_names= microbe_names[microbe_idx[i]].reshape(-1, 1)
		rank= np.arange(1, topk+ 1, 1).reshape(-1, 1)
		probability= microbe_result[i][microbe_idx[i]].reshape(-1, 1)
		ls.append(np.concatenate((di_names, rank, di_microbe_names, probability), axis= 1))
	drug_i_idx= 1343
	result= np.concatenate(ls, axis= 0)
	np.savetxt(f'./run_epoch/{epoch}/candidate_microbes.txt', result, fmt= '%s', delimiter= '\t', encoding= 'utf-8')

# top candidate ls
def topk_candidate_microbles(epoch, topk= 20):

	microbe_names= []
	with open('../data/microbe_names.txt', 'r', encoding= 'utf-8') as f:
		for line in f.readlines():
			microbe_names.append(line.rstrip())
	microbe_result= []
	microbe_result= torch.from_numpy(np.loadtxt(f'./epoch/{epoch}epoch/case_analysis_result_{epoch}.txt'))
	microbe_idx= microbe_result.sort(dim= 1, descending= True)[1][:, 0: topk].to(torch.long)
	for i in range(topk):
		print(f'CIPROFLOXACIN top {i+ 1} candidate microbe: {microbe_names[microbe_idx[0, i]]}; Moxifloxacin top {i+ 1} candidate microbe: {microbe_names[microbe_idx[1, i]]}; Vancomycin top {i+ 1} candidate microbe: {microbe_names[microbe_idx[2, i]]};')

# microbes related with ciprofloxacin and Moxifloxacin
def ls_candidate_microbles_name(database, drug_name, dir1= 'E:/data'):
	
	drug_names, microbe_names= [], []
	with open(os.path.join(dir1, database, 'drugs.txt'), 'r', encoding= 'utf-8') as f:
		for line in f.readlines():
			drug_names.append(line.rstrip())
	with open(os.path.join(dir1, database, 'microbes.txt'), 'r', encoding= 'utf-8') as f:
		for line in f.readlines():
			microbe_names.append(line.rstrip())
	drug_idx= drug_names.index(drug_name)
	adj_info= np.loadtxt(os.path.join(dir1, database, 'adj.txt'))
	col1, col2, col3= adj_info[:, 0], adj_info[:, 1], adj_info[:, 2]
	adj_mat= coo_matrix((col3, (col1- 1, col2- 1)), shape= (len(drug_names), len(microbe_names)), dtype= np.int).toarray()
	microbe_candidate_idxs= np.where(adj_mat[drug_idx, :]== 1)[0]
	print(f'in {database}, {drug_name} association: ')
	for idx in microbe_candidate_idxs:
		print(microbe_names[idx])
	print('\n')
