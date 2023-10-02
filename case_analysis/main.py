import os
import dgl
import time
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
from model import NGMDA
import torch.nn.functional as F
from data_loader import dataloader

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
	xy4train, label4train, xy4test= dl.xy4train, dl.label4train, dl.xy4test
	loss_func1, loss_func2= nn.CrossEntropyLoss(), nn.MSELoss()
	n, pos_num= len(label4train), label4train.sum()
	train_xy_label_tuple_dataset= torch.utils.data.TensorDataset(xy4train, label4train)
	train_loader= torch.utils.data.DataLoader(train_xy_label_tuple_dataset, batch_size= args.batch_size, shuffle= False)
	fea_mats= [dl.drug_sim_mat, dl.micro_sim_mat, dl.links['ass_mat'][0], dl.links['ass_mat'][2], dl.links['ass_mat'][1]]
	fea_mats= [fea_mat.to(args.device) for fea_mat in fea_mats]
	adj_mat=torch.cat((torch.cat((dl.links['ass_mat'][0], dl.links['ass_mat'][1]), dim= 1), torch.cat((dl.links['ass_mat'][1].T, dl.links['ass_mat'][2]), dim= 1)), dim= 0)
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
	e_feat= torch.tensor(e_feat, dtype= torch.long).to(args.device)
	head4gat, head4tf= args.head4gat, args.head4transformer
	net= NGMDA(g, fea_mats, e_feat, args.hd4gat, args.hd4tf, args.layers_num, head4gat, head4tf, args.feat_dropout, args.attn_dropout, args.wr_time, args.decoder).to(args.device)
	epoch= 119
	# net.load_state_dict(torch.load('checkpoint_epoch59'))	
	# optimizer= torch.optim.Adam(net.parameters(), lr= args.lr, weight_decay= args.weight_decay)
	# net.train()
	# for epoch in range(args.epochs):
		# for step, (x_y, x_y_label) in enumerate(train_loader):
			# net.train()
			# t_start= time.time()
			# x_y_label, x_y= x_y_label.to(args.device), x_y.to(args.device)
			# train_loss2, logp= net(x_y[:, 0], x_y[:, 1]+ y_offset, topo_fea, loss_func2, loss_func1)
			# train_loss1= loss_func1(logp, x_y_label)
			# train_loss= 0.8* train_loss1+  0.2* train_loss2
			# optimizer.zero_grad()
			# train_loss.backward()
			# optimizer.step()
			# correct_num= (torch.max(logp, dim= 1)[1]== x_y_label).float().sum()
			# t_end= time.time()
			# print(f'epoch: {epoch+ 1}, step: {step+ 1}, train loss: {train_loss1.item()}, {train_loss2}, time: {t_end- t_start}, acc: {correct_num/len(x_y_label)}')
		# torch.save(net.state_dict(), f'checkpoint_epoch{epoch+ begin}')
		# print(f'epoch{epoch} run down!!!')
	net.load_state_dict(torch.load(f'./run_epoch/{epoch}/checkpoint_epoch{epoch}'))
	print(f'testing...')
	net.eval()
	CAR= []
	with torch.no_grad():
		for i in range(math.ceil(len(xy4test)/ args.batch_size* 1.0)):
			batch_xy4test= xy4test[(i)* args.batch_size: (i+ 1)* args.batch_size].to(args.device)
			_, logp= net(batch_xy4test[:, 0], batch_xy4test[:, 1]+ y_offset, topo_fea, loss_func2, loss_func1)
			logp= F.softmax(logp, dim= 1)
			CAR.append(logp[:, 1])
			print(f'batch{i}/ {math.ceil(1373* 173/ args.batch_size* 1.0)} finshed!!!')
	CAR= torch.cat(CAR, dim= 0).view(1373, 173)
	np.savetxt(f'./run_epoch/{epoch}/case_analysis_result_all_drugs.txt', CAR.cpu().numpy())

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
	ap.add_argument('--weight_decay', type= float, default= 5e-4)
	ap.add_argument('--patience', type= int, default= 90)
	ap.add_argument('--epochs', type= int, default= 60)
	ap.add_argument('--batch_size', type= int, default= 20000)
	ap.add_argument('--kflod_num', type= int, default= 5)
	ap.add_argument('--lbda', type= int, default= 4)
	ap.add_argument('--wr_time', type= int, default= 2)
	ap.add_argument('--i', type= int, default= 0)
	args= ap.parse_args([])
	run_model(args)