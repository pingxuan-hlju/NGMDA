import os
import math
import torch
import numpy as np
from scipy.io import loadmat
import torch.nn.functional as F


class dataloader:
	def __init__(self, root_path):
		torch.manual_seed(1206)
		self.file_name= 'data'
		self.ass_file= 'net1.mat'
		self.root_path= root_path
		self.drug_sim_file= 'drugsimilarity.txt'
		self.micro_sim_file= 'micr_cosine_sim.txt'
		# microbe_microbe_similarity
		self.threshold4drug= 0.9
		self.threshold4micro= 0.999
		self.links= self.load_links()
		self.nodes= self.load_nodes()
		self.drug_sim_mat, self.micro_sim_mat= self.load_sim_mat()
		self.drug_feat01_mat, self.micro_feat01_mat= torch.eye(1373), torch.eye(173)
		# train set, test set.
		self.xy4train, self.label4train, self.xy4test= self.gen_pos_neg_xy()

	# @ check
	def load_sim_mat(self):
		return torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.drug_sim_file))).to(torch.float),\
		torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.micro_sim_file))).to(torch.float)

	# @  train set, test set,
	def gen_pos_neg_xy(self):
		ass_mat, pos_num= self.links['ass_mat'][1], 2470
		n = ass_mat.shape[0]* ass_mat.shape[1]
		rand_num_237529= torch.randperm(n)
		neg_xy, pos_xy= (ass_mat== 0).nonzero(), ass_mat.nonzero()
		xy4train, label4train= torch.cat((neg_xy, pos_xy), dim= 0)[rand_num_237529], torch.cat((torch.zeros(n- 2470), torch.ones(2470)), dim= 0)[rand_num_237529]
		xy4test= torch.cartesian_prod(torch.range(0, 1372).to(torch.long), torch.range(0, 172).to(torch.long))
		return xy4train, label4train.to(torch.long), xy4test

	# outs, pred label; labels, true label; test_xy;
	def outs2file(self, fold, preds, labels, test_xy):
		eva_labels_outs_x_y= np.zeros((len(test_xy), 4))
		for i in range(len(test_xy)):
			eva_labels_outs_x_y[i, 2], eva_labels_outs_x_y[i, 3], = test_xy[i, 0], test_xy[i, 1]
			eva_labels_outs_x_y[i, 0]= labels[i]
			eva_labels_outs_x_y[i, 1]= preds[i]
		np.savetxt(f'./fold{fold}.txt', eva_labels_outs_x_y)

	# @ check
	def load_nodes(self):
		nodes= {'total': 0, 'type': {}, 'type_num': {}, 'feature': {}}
		nodes['type']= {'drug': 0, 'micro': 1}
		nodes['total']= 1546
		nodes['type_num']= {0: 1373, 1: 173}
		nodes['feature']= {0: torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.drug_sim_file))).to(torch.float), 1: torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.micro_sim_file))).to(torch.float)}
		return nodes

	# @ check
	def load_links(self):
		drug_micro_mat= loadmat(os.path.join(self.root_path, self.file_name, self.ass_file), mat_dtype= True)['interaction']
		links= {'total': 0, 'type': {}, 'link_num': {}, 'ass_mat': {}, 'sim_mat': {}}
		drug_sim, micro_sim= torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.drug_sim_file))), torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.micro_sim_file)))
		drug_sim[(drug_sim>= self.threshold4drug)], micro_sim[micro_sim>= self.threshold4micro]= 1, 1
		drug_sim[drug_sim< self.threshold4drug], micro_sim[micro_sim< self.threshold4micro]= 0, 0
		links['sim_mat']= {0: drug_sim, 1: micro_sim}
		links['type']= {'d2d': 0, 'd2m': 1, 'm2m': 2}
		links['ass_mat']= {0: drug_sim, 1: torch.from_numpy(drug_micro_mat).to(torch.float), 2: micro_sim}
		links['link_num']= {0: drug_sim.sum(), 1: drug_micro_mat.sum(), 2: micro_sim.sum()}
		links['total']= links['link_num'][0]+ links['link_num'][1]+ links['link_num'][2]
		return links

	# @ tensor shuffle
	def tensor_shuffle(self, ts, dim= 0):
		return ts[torch.randperm(ts.shape[dim])]

	# write file
	def gen_file_for_args(self, args, test_loss, acc, auc, aupr, file_path= os.path.abspath('..')+ '/eva/eva.txt'):
		with open(file_path, 'a') as f:
			f.write(f'{args.lr}\t{args.weight_decay}\t{test_loss}\t{args.batch_size}\t{acc}\t{auc}\t{aupr}\n')

	# @ cosine simi
	def cosine_sim(self, ts):
		return 0.5* (torch.matmul(F.normalize(ts, p= 2, dim= 1), F.normalize(ts, p= 2, dim= 1).T)+ 1)

