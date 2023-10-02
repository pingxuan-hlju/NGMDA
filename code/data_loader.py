import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat


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
		self.pos_neg_xy, self.pos_neg_label, self.rest_neg_xy, self.rest_neg_label= self.gen_pos_neg_xy()

	# @
	def load_sim_mat(self):
		return torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.drug_sim_file))).to(torch.float),\
		torch.from_numpy(np.loadtxt(os.path.join(self.root_path, self.file_name, self.micro_sim_file))).to(torch.float)

	# @ obtain train set, test set
	def gen_pos_neg_xy(self):
		ass_mat, rand_num_4940 = self.links['ass_mat'][1], torch.randperm(4940)
		pos_xy= ass_mat.nonzero()
		pos_label= torch.ones(len(pos_xy))
		neg_xy= self.tensor_shuffle((ass_mat== 0).nonzero(), dim= 0)
		neg_xy, rest_neg_xy, neg_label, rest_neg_label= neg_xy[0: len(pos_xy)], neg_xy[len(pos_xy):], torch.zeros(len(pos_xy)), torch.zeros(len(neg_xy)- len(pos_xy))
		pos_neg_xy, pos_neg_label= torch.cat((pos_xy, neg_xy), dim= 0)[rand_num_4940], torch.cat((pos_label, neg_label), dim= 0)[rand_num_4940]
		return pos_neg_xy, pos_neg_label.to(torch.long), rest_neg_xy, rest_neg_label.to(torch.long)

	# outs, pred value; labels, true label; test_xy;
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

	# @ shuffle
	def tensor_shuffle(self, ts, dim= 0):
		return ts[torch.randperm(ts.shape[dim])]

	# write file
	def gen_file_for_args(self, args, test_loss, acc, auc, aupr, file_path= os.path.abspath('..')+ '/eva/eva.txt'):
		with open(file_path, 'a') as f:
			f.write(f'{args.lr}\t{args.weight_decay}\t{test_loss}\t{args.batch_size}\t{acc}\t{auc}\t{aupr}\n')

	# @ cpt simi
	def cosine_sim(self, ts):
		return 0.5* (torch.matmul(F.normalize(ts, p= 2, dim= 1), F.normalize(ts, p= 2, dim= 1).T)+ 1)
