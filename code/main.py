import os
import dgl
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
from model import NGMDA
from tools import EarlyStopping
import torch.nn.functional as F
from data_loader import dataloader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve


def acquire_topo_fea(ass_mat, k):
	topo_fea= torch.zeros((1546, k))
	d_mat= torch.diag(ass_mat.sum(dim= 1)).inverse()
	topo_mat= torch.matmul(ass_mat, d_mat)
	for i in range(k):
		topo_fea[:, i]= torch.diag(topo_mat)
		topo_mat= torch.matmul(topo_mat, topo_mat)
	return topo_fea

def run_model(args):

	torch.manual_seed(1206)
	dl= dataloader(args.root_path)
	pos_neg_xy, pos_neg_label, rest_neg_xy, rest_neg_label= dl.pos_neg_xy, dl.pos_neg_label, dl.rest_neg_xy, dl.rest_neg_label
	loss_func1, loss_func2, kflod= nn.CrossEntropyLoss(), nn.MSELoss(), KFold(n_splits= args.kflod_num, shuffle= False)

	for fold, (train_xy_idx, test_xy_idx) in enumerate(kflod.split(pos_neg_xy)):
		print(f'{fold+ 1} fold, pos num in training: {pos_neg_label[train_xy_idx].sum()}, neg num in trainning: {len(train_xy_idx)- pos_neg_label[train_xy_idx].sum()}')		

		test_xy_bal, test_label0_bal= pos_neg_xy[test_xy_idx, ], pos_neg_label[test_xy_idx, ]
		test_xy, test_label= torch.cat((test_xy_bal, rest_neg_xy), dim= 0).to(args.device), torch.cat((test_label0_bal, rest_neg_label), dim= 0).to(args.device)
		test_xy_bal, test_label0_bal= test_xy_bal.to(args.device), test_label0_bal.to(args.device)
		# package
		train_xy_label_tuple_dataset= torch.utils.data.TensorDataset(pos_neg_xy[train_xy_idx, ], pos_neg_label[train_xy_idx, ])
		test_xy_label_tuple_dataset= torch.utils.data.TensorDataset(test_xy, test_label)
		train_loader= torch.utils.data.DataLoader(train_xy_label_tuple_dataset, batch_size= args.batch_size, shuffle= False)
		test_loader= torch.utils.data.DataLoader(test_xy_label_tuple_dataset, batch_size= 2* args.batch_size, shuffle= False)
		# update ass_mat
		ass_mat= dl.links['ass_mat'][1].clone()
		ass_mat[test_xy[:, 0], test_xy[:, 1]]= 0
		# 
		fea_mats= [dl.drug_sim_mat, dl.micro_sim_mat, dl.links['ass_mat'][0], dl.links['ass_mat'][2], ass_mat]
		fea_mats= [fea_mat.to(args.device) for fea_mat in fea_mats]
		# obtain heterogeneous graph; k: WR step; lambda, lbda: eps;
		adj_mat=torch.cat((torch.cat((dl.links['ass_mat'][0], ass_mat), dim= 1), torch.cat((ass_mat.T, dl.links['ass_mat'][2]), dim= 1)), dim= 0)
		# topo feature
		topo_fea= acquire_topo_fea(adj_mat, args.wr_time).to(args.device)
		x_y= adj_mat.nonzero()
		x, y, y_offset= x_y[:, 0], x_y[:, 1], torch.tensor(dl.nodes['type_num'][0]).to(args.device)
		adj= sp.coo_matrix((torch.ones(len(x)), (x, y)), shape= (dl.nodes['total'], dl.nodes['total'])).tocsr()
		g= dgl.DGLGraph(adj)
		g= dgl.add_self_loop(g)
		e_feat= []
		for u, v in zip(*g.edges()):
			u, v= u.item(), v.item()
			if u< 1373 and v< 1373:
				e_type= 0
			elif u< 1373 and v>= 1373:
				e_type= 1
			elif u>= 1373 and v< 1373:
				e_type= 2
			else: 
				e_type= 3
			e_feat.append(e_type)
		g= g.to(args.device)
		# edge type infomation
		e_feat= torch.tensor(e_feat, dtype= torch.long).to(args.device)
		head4gat, head4tf= args.head4gat, args.head4transformer
		# define model
		net= NGMDA(g, fea_mats, e_feat, args.hd4gat, args.hd4tf, args.layers_num, head4gat, head4tf, args.feat_dropout, args.attn_dropout, args.wr_time, args.decoder).to(args.device)
		optimizer= torch.optim.Adam(net.parameters(), lr= args.lr, weight_decay= args.weight_decay)
		net.train()
		early_stopping= EarlyStopping(patience= args.patience, verbose= True, save_path= os.path.join(args.root_path, 'pt', 'checkpoint.pt'))
		# batch
		for epoch in range(args.epochs):
			for step, (x_y, x_y_label) in enumerate(train_loader):
				net.train()
				t_start= time.time()
				x_y_label, x_y= x_y_label.to(args.device), x_y.to(args.device)
				train_loss2, logp= net(x_y[:, 0], x_y[:, 1]+ y_offset, topo_fea, loss_func2, loss_func1)
				train_loss1= loss_func1(logp, x_y_label)
				train_loss= (1- args.lbda)* train_loss1+  args.lbda* train_loss2
				optimizer.zero_grad()
				train_loss.backward()
				optimizer.step()
				correct_num= (torch.max(logp, dim= 1)[1]== x_y_label).float().sum()
				t_end= time.time()
				print(f'epoch: {epoch+ 1}, step: {step+ 1}, train loss: {train_loss1.item()}, {train_loss2}, time: {t_end- t_start}, acc: {correct_num/len(x_y_label)}')
				# eval
				t_start= time.time()
				net.eval()
				with torch.no_grad():
					val_loss2, logp= net(test_xy_bal[:, 0], test_xy_bal[:, 1]+ y_offset, topo_fea, loss_func2, loss_func1)
					correct_num= (torch.max(logp, dim= 1)[1]== test_label0_bal).float().sum()
					val_loss1= loss_func1(logp, test_label0_bal)
					val_loss= val_loss1+ val_loss2
				t_end= time.time()
				print(f'epoch: {epoch+ 1}, val loss: {val_loss1.item()}, {val_loss2}, time: {t_end- t_start}, acc: {correct_num/len(test_label0_bal)}')
				early_stopping(val_loss, net)
				if early_stopping.early_stop:
					print(f'early_stopping!')
					break
			if early_stopping.early_stop:
				print(f'early_stopping!')
				break
		net.load_state_dict(torch.load(os.path.join(args.root_path, 'pt', 'checkpoint.pt')))
		net.eval()
		with torch.no_grad():
			pred_ls= []
			for step, (x_y, x_y_label) in enumerate(test_loader):
				_, logp= net(x_y[:, 0], x_y[:, 1]+ y_offset, topo_fea, loss_func2, loss_func1)
				pred_ls.append(logp)
			preds= torch.cat(pred_ls, dim= 0)
			loss1= loss_func1(preds, test_label)
			acc= (torch.max(preds, dim= 1)[1]== test_label).float().sum()/(len(test_label))
			precision, recall, threshold= precision_recall_curve(test_label.cpu(), preds[:, 1].cpu())
			roc_auc, aupr= roc_auc_score(test_label.cpu(), preds[:, 1].cpu()), auc(recall, precision)
			print(f'acc: {acc}, roc_auc: {roc_auc}, aupr: {aupr}')
			dl.gen_file_for_args(args, loss1, acc, roc_auc, aupr, file_path= args.root_path+ '/eva/eva.txt')
			dl.outs2file(fold, preds[:, 1].cpu(), test_label.cpu(), test_xy.cpu())


if __name__== '__main__':
	ap= argparse.ArgumentParser(description= 'NGMDA')
	ap.add_argument('--root-path', type= str, default= os.path.abspath('..'))
	ap.add_argument('--device', type= str, default= 'cuda:0')
	ap.add_argument('--hd4gat', type= int, default= 128)
	ap.add_argument('--hd4tf', type= int, default= 64)
	ap.add_argument('--layers_num', type= int, default= 2)
	ap.add_argument('--head4gat', type= int, default= 4)
	ap.add_argument('--head4transformer', type= int, default= 8)
	ap.add_argument('--feat_dropout', type= float, default= 0.5)
	ap.add_argument('--attn_dropout', type= float, default= 0.5)
	ap.add_argument('--decoder', type= str, default= 'mlp')
	ap.add_argument('--lr', type= float, default= 1e-4)
	ap.add_argument('--weight_decay', type= float, default= 5e-5)
	ap.add_argument('--patience', type= int, default= 90)
	ap.add_argument('--epochs', type= int, default= 100)
	ap.add_argument('--batch_size', type= int, default= 128)
	ap.add_argument('--kflod_num', type= int, default= 5)
	# balance item
	ap.add_argument('--lbda', type= float, default= 0.2)
	# random walk step
	ap.add_argument('--wr_time', type= int, default= 2)
	ap.add_argument('--i', type= int, default= 0)
	args= ap.parse_args([])
	run_model(args)