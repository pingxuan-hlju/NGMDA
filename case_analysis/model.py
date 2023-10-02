import dgl
import math
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import scipy.sparse as sp
from dgl.nn import GraphConv
import torch.nn.functional as F
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import edge_softmax
from sklearn.metrics.pairwise import cosine_similarity

# gat layer.
class HGNConv(nn.Module):
	def __init__(self, in_feats, out_feats, num_heads, feat_drop= 0., attn_drop= 0.):
		super(HGNConv, self).__init__()
		self._num_heads, self._out_feats, self.node_type_num= num_heads, out_feats, 2
		self.fc4node_proj= nn.ModuleList([nn.Linear(in_feats, out_feats* num_heads, bias= False) for i in range(self.node_type_num)])
		self.fc4prob= nn.ModuleList([nn.Linear(out_feats, out_feats, bias= False) for i in range(self._num_heads)])
		# edge type attn: d2d, d2m, m2d, m2m
		self.etype_attn= nn.Parameter(torch.tensor([1, 1, 1, 1]).to(torch.float))
		self.feat_drop4cont, self.feat_drop4loc, self.attn_drop= nn.Dropout(feat_drop), nn.Dropout(feat_drop), nn.Dropout(attn_drop)
		# fc 4 residual
		self.fc4res= nn.ModuleList([nn.Linear(in_feats, out_feats* num_heads, bias= False) for i in range(self.node_type_num)])
		self.relu, self.tao= nn.ReLU(), 0.4
		self.reset_parameters()

	def reset_parameters(self):
		for fc in self.fc4node_proj: nn.init.xavier_normal_(fc.weight)
		for fc in self.fc4res: nn.init.xavier_normal_(fc.weight)
		for fc in self.fc4prob: nn.init.xavier_normal_(fc.weight, gain= nn.init.calculate_gain('tanh'))

	# cpt attn
	def edge_attention(self, edges):
		src_feat_nrom, src_sloc, dst_sloc, src_tp_fea, dst_tp_fea= edges.src['ft_norm'], edges.src['sloc'], edges.dst['sloc'], edges.src['topo_fea'], edges.dst['topo_fea']
		a= torch.mul(edges.dst['ft_attn'], src_feat_nrom).sum(dim= 2, keepdim= True)
		sloc_sim= ((self.tao* self.cosine(src_sloc, dst_sloc)+ (1- self.tao)* self.cosine(src_tp_fea, dst_tp_fea))+ 1)/ 2
		# sloc_sim= 1
		return {'e': (sloc_sim* a)* edges.data['e_type_attn'].unsqueeze(-1).unsqueeze(-1).repeat(1, a.shape[1], 1)}

	# cosine simi
	def cosine(self, a, b):
		return torch.mul(F.normalize(a, p= 2, dim= 2), F.normalize(b, p= 2, dim= 2)).sum(dim= 2, keepdim= True)

	def forward(self, graph, feat, sloc, topo_fea):
		with graph.local_scope():
			sloc= self.feat_drop4loc(sloc.unsqueeze(1))
			topo_fea= self.feat_drop4loc(topo_fea.unsqueeze(1))
			h_src= h_dst= self.feat_drop4cont(feat)
			# feat_src, feat_dst, (1546, 4, 16);
			feat_src= feat_dst= torch.cat([self.fc4node_proj[0](h_src[0: 1373, :]), self.fc4node_proj[1](h_src[1373: , :])], dim= 0).view(-1, self._num_heads, self._out_feats)
			# (1546, 4, 16) >> [(1546, 16), ..., ] >> (1546, 64) >> (1546, 4, 16); 
			feat_dst_attn= torch.cat([F.softmax(torch.tanh(self.fc4prob[i](feat_dst[:, i, :])), dim= 1) for i in range(self._num_heads)], dim= 1).view(-1, self._num_heads, self._out_feats)
			graph.edata.update({'e_type_attn': self.etype_attn[graph.edata['e_feat']]})
			# ft, (1546, 4, 16)按dim= 0, L2;
			graph.srcdata.update({'ft_norm': F.normalize(feat_src, p= 2, dim= 0), 'ft': feat_src, 'sloc': sloc, 'topo_fea': topo_fea})
			# ft_attn, (1546, 4, 16);
			graph.dstdata.update({'ft_attn': feat_dst_attn, 'sloc': sloc, 'topo_fea': topo_fea})
			# update edge attention
			graph.apply_edges(self.edge_attention)
			graph.edata['a']= self.attn_drop(edge_softmax(graph, graph.edata.pop('e')))
			# mess pass
			graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
			graph.update_all(fn.u_mul_e('sloc', 'a', 'm2'), fn.sum('m2', 'sloc'))
			graph.update_all(fn.u_mul_e('topo_fea', 'a', 'm3'), fn.sum('m3', 'topo_fea'))
			# update
			# rst, (1546, 4, 16);
			rst= graph.dstdata['ft']
			sloc2= graph.dstdata['sloc']
			topo_fea2= graph.dstdata['topo_fea']
			# residual
			resval= torch.cat([self.fc4res[0](h_dst[0: 1373]), self.fc4res[1](h_dst[1373:])], dim= 0).view(h_dst.shape[0], -1, self._out_feats)
			# (1546, 4, 64); -->> (1546, 4, 64);
			restult_fea= self.relu(rst+ resval).mean(1)
			sloc2= (sloc2+ sloc).mean(1)
			topo_fea= (topo_fea2+ topo_fea).mean(1)
			return restult_fea, sloc2, topo_fea

# transformer block
class block_tf(nn.Module):
	def __init__(self, node_dim= 64, edge_dim= 64, heads= 2):
		super().__init__()
		self.node_dim, self.edge_dim, self.heads= node_dim, edge_dim, heads
		# Q, K, V , (128, 64)
		[self.Wq1, self.Wk1, self.Wv1]= [nn.Linear(self.node_dim+ self.edge_dim, self.node_dim) for i in range(3)]
		[self.Wq2, self.Wk2, self.Wv2]= [nn.Linear(self.node_dim+ self.edge_dim, self.node_dim) for i in range(3)]		
		self.W1, self.W2, self.W3= nn.Linear(self.node_dim, self.node_dim), nn.Linear(self.node_dim, self.node_dim), nn.Linear(self.node_dim, self.node_dim)
		self.relu= nn.ReLU()
		self.layer_norm= nn.LayerNorm(self.node_dim)
		# init parameters
		self.reset_parameters()

	def reset_parameters(self):	
		nn.init.xavier_normal_(self.Wq1.weight)
		nn.init.xavier_normal_(self.Wk1.weight)
		nn.init.xavier_normal_(self.Wv1.weight)
		nn.init.xavier_normal_(self.Wq2.weight)
		nn.init.xavier_normal_(self.Wk2.weight)
		nn.init.xavier_normal_(self.Wv2.weight)
		nn.init.xavier_normal_(self.W1.weight)
		nn.init.xavier_normal_(self.W2.weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.W3.weight)		

	def forward(self, node_mat, edge_type_mat):
		drug_mat, micro_mat, idx0, idx1, idx2, idx3= node_mat[0: 1373], node_mat[1373:], torch.tensor([0]).cuda(), torch.tensor([1]).cuda(), torch.tensor([2]).cuda(), torch.tensor([3]).cuda()
		# rel0, d2d, rel1, d2m, rel2, m2d, rel3, m2m;
		h1= torch.cat((drug_mat, edge_type_mat(idx0).repeat(drug_mat.shape[0], 1)), dim= 1)
		# (1373, 64)>> (1373, heads, 64// heads)>> (heads, 1373, 64// heads)
		Q1= self.Wq1(h1).view(h1.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		# (1373, 64)>> (1373, heads, 64// heads)>> (heads, 1373, 64// heads)
		K1= self.Wk1(h1).view(h1.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		# (1373, 64)>> (1373, heads, 64// heads)>> (heads, 1373, 64// heads)
		V1= self.Wv1(h1).view(h1.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		h2i, h2j= torch.cat((drug_mat, edge_type_mat(idx2).repeat(drug_mat.shape[0], 1)), dim= 1), torch.cat((micro_mat, edge_type_mat(idx2).repeat(micro_mat.shape[0], 1)), dim= 1)
		# (heads, 1373, 64// heads)
		Q2= self.Wq1(h2i).view(h2i.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		# (heads, 173, 64// heads)
		K2= self.Wk2(h2j).view(h2j.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		# (heads, 173, 64// heads)
		V2= self.Wv2(h2j).view(h2j.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		# (heads, 1373, 64// heads), (heads, 1373, 64// heads)>> (4, 1373, 1373); (heads, 1373, 64// heads), (heads, 173, 64// heads)>> (4, 1373, 173); (4, 1373, 1546) 
		att4drug_kij_hat= torch.cat((torch.matmul(Q1, K1.transpose(1, 2))/ math.sqrt(Q1.shape[-1]), torch.matmul(Q2, K2.transpose(1, 2))/ math.sqrt(Q2.shape[-1])), dim= 2)
		# (4, 1373, 1546),>> (4, 1373, 1546); softmax by neighbors
		att4drug_kij= F.softmax(att4drug_kij_hat, dim= 2)
		# (4, 1373, 1546), (4, 1546, 32)>> (4, 1373, 32)>> (1373, 4, 32)>> (1373, 128)
		mess1= torch.matmul(att4drug_kij, torch.cat((V1, V2), dim= 1)).transpose(0, 1).reshape(drug_mat.shape[0], -1)
		# fuse, (1373, 128)
		cofusion4drug= self.layer_norm(self.W1(mess1)+ drug_mat)
		# obtain new feature
		drug_mat_new= self.layer_norm(self.W3(self.relu(self.W2(cofusion4drug)))+ cofusion4drug)
		# 
		h3i, h3j= torch.cat((micro_mat, edge_type_mat(idx1).repeat(micro_mat.shape[0], 1)), dim= 1), torch.cat((drug_mat, edge_type_mat(idx1).repeat(drug_mat.shape[0], 1)), dim= 1)
		Q3= self.Wq2(h3i).view(h3i.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		K3= self.Wk1(h3j).view(h3j.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		V3= self.Wv1(h3j).view(h3j.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)		
		h4= torch.cat((micro_mat, edge_type_mat(idx3).repeat(micro_mat.shape[0], 1)), dim= 1)
		Q4= self.Wq2(h4).view(h4.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		K4= self.Wk2(h4).view(h4.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)
		V4= self.Wv2(h4).view(h4.shape[0], self.heads, (self.node_dim)// self.heads).transpose(0, 1)				
		att4micro_kij_hat= torch.cat((torch.matmul(Q3, K3.transpose(1, 2))/ math.sqrt(Q3.shape[-1]), torch.matmul(Q4, K4.transpose(1, 2))/ math.sqrt(Q4.shape[-1])), dim= 2)
		att4micro_kij= F.softmax(att4micro_kij_hat, dim= 2)
		mess2= torch.matmul(att4micro_kij, torch.cat((V3, V4), dim= 1)).transpose(0, 1).reshape(micro_mat.shape[0], -1)
		cofusion4micro= self.layer_norm(self.W1(mess2)+ micro_mat)
		micro_mat_new= self.layer_norm(self.W3(self.relu(self.W2(cofusion4micro)))+ cofusion4micro)
		# (1546, 64)
		return torch.cat((drug_mat_new, micro_mat_new), dim= 0)

# @
class Decoder4Path(nn.Module):

	def __init__(self, in_dim):
		super().__init__()
		self.seq4BOut= nn.Sequential(nn.Linear(2* in_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5))
		self.out= nn.Linear(256, 2)
		self.reset_parameters()
	def reset_parameters(self):
		for mode in self.seq4BOut:
			if isinstance(mode, nn.Linear):
				nn.init.xavier_normal_(mode.weight, gain= nn.init.calculate_gain('relu'))		
		nn.init.xavier_normal_(self.out.weight)
	def forward(self, left_emb, right_emb):
		return self.out(self.seq4BOut(torch.cat((left_emb, right_emb), dim= 1)))

# @
class Decoder(nn.Module):
	def __init__(self, in_dim1, in_dim2):
		super().__init__()
		self.seq41= nn.Sequential(nn.Linear(in_dim1* 2, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5))
		self.seq42= nn.Sequential(nn.Linear(in_dim2* 2, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 64), nn.ReLU(), nn.Dropout(0.5))
		self.seq4ori= nn.Sequential(nn.Linear(1546* 2, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5))
		self.seq4out= nn.Sequential(nn.Linear(448, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))
		self.reset_para()
	
	def reset_para(self):
		for mode in self.seq41:
			if isinstance(mode, nn.Linear):
				nn.init.xavier_normal_(mode.weight, gain= nn.init.calculate_gain('relu'))
		for mode in self.seq42:
			if isinstance(mode, nn.Linear):
				nn.init.xavier_normal_(mode.weight, gain= nn.init.calculate_gain('relu'))
		for mode in self.seq4ori:
			if isinstance(mode, nn.Linear):
				nn.init.xavier_normal_(mode.weight, gain= nn.init.calculate_gain('relu'))				
		nn.init.xavier_normal_(self.seq4out[0].weight, gain= nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.seq4out[3].weight)

	def forward(self, left_emb1, right_emb1, left_emb2, right_emb2):
		left_ori, right_ori= left_emb1[:, 0: 1546]+ left_emb2[:, 0: 1546], right_emb1[:, 0: 1546]+ right_emb2[:, 0: 1546]
		ori_out= self.seq4ori(torch.cat((left_ori, right_ori), dim= 1))
		out1, out2= self.seq41(torch.cat((left_emb1[:, 1546:], right_emb1[:, 1546: ]), dim= 1)), self.seq42(torch.cat((left_emb2[:, 1546: ], right_emb2[:, 1546: ]), dim= 1))
		return self.seq4out(torch.cat((ori_out, out1, out2), dim= 1))

# ensemble heterogeneous model for microbe and drug associate prediction.
class NGMDA(nn.Module):

	def __init__(self, g, features_list, e_feat, hd4gat, hd4tf, num_layers, head4gat, head4tf, feat_drop, attn_drop, k, decoder= 'mlp'):
		super(NGMDA, self).__init__()
		self.g= g
		self.g.edata['e_feat']= e_feat
		self.features_list= features_list
		self.gat_layers= nn.ModuleList()
		self.num_layers= num_layers
		# AE 4 GAT
		self.encoder4sloc= nn.Sequential(nn.Linear(1546, 512), nn.ReLU(), nn.Linear(512, hd4gat), nn.ReLU())
		self.decoder4sloc= nn.Sequential(nn.Linear(hd4gat, 512), nn.ReLU(), nn.Linear(512, 1546), nn.Sigmoid())			
		self.encoder4micro= nn.Sequential(nn.Linear(173, hd4gat), nn.ReLU())
		self.decoder4micro= nn.Sequential(nn.Linear(hd4gat, 173), nn.Sigmoid())
		self.encoder4drug= nn.Sequential(nn.Linear(1373, 512), nn.ReLU(), nn.Linear(512, hd4gat), nn.ReLU())
		self.decoder4drug= nn.Sequential(nn.Linear(hd4gat, 512), nn.ReLU(), nn.Linear(512, 1373), nn.Sigmoid())
		self.decoder4classify= nn.Linear(hd4gat, 2)
		# AE 4 transformer
		self.encoder4drug2= nn.Sequential(nn.Linear(1546, 512), nn.ReLU(), nn.Linear(512, hd4tf), nn.ReLU())
		self.encoder4micro2= nn.Sequential(nn.Linear(1546, 512), nn.ReLU(), nn.Linear(512, hd4tf), nn.ReLU())
		self.decoder4drug2= nn.Sequential(nn.Linear(hd4tf, 512), nn.ReLU(), nn.Linear(512, 1546), nn.Sigmoid())			
		self.decoder4micro2= nn.Sequential(nn.Linear(hd4tf, 512), nn.ReLU(), nn.Linear(512, 1546), nn.Sigmoid())
		self.decoder4classify2= nn.Linear(hd4tf, 2)
		# edge type embedding 
		self.edge_type_emb= nn.Embedding(4, hd4tf)
		# transformer layers
		self.tf_layers= nn.ModuleList([block_tf(node_dim= hd4tf, edge_dim= hd4tf, heads= head4tf) for i in range(num_layers)])
		# gat layers
		for l in range(0, num_layers):
			self.gat_layers.append(HGNConv(hd4gat, hd4gat, head4gat, feat_drop, attn_drop))
		# Decoder layers
		self.decoder= Decoder((hd4gat* 2+ 2)* (num_layers+ 1), (hd4tf)* (num_layers+ 1))
		# self.decoder= Decoder4Path(1546+ (hd4tf)* (num_layers+ 1))
		# init parameters
		self.reset_parameters()

	# init weight
	def reset_parameters(self):
		# encoder& decoder4 gat
		nn.init.xavier_normal_(self.encoder4sloc[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.encoder4sloc[2].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4sloc[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4sloc[2].weight, nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.encoder4drug[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.encoder4drug[2].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4drug[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4drug[2].weight, nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.encoder4micro[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4micro[0].weight, nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.decoder4classify.weight)
		# encoder& decoder4 transformer
		nn.init.xavier_normal_(self.encoder4drug2[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.encoder4drug2[2].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4drug2[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4drug2[2].weight, nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.encoder4micro2[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.encoder4micro2[2].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4micro2[0].weight, nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.decoder4micro2[2].weight, nn.init.calculate_gain('sigmoid'))
		nn.init.xavier_normal_(self.decoder4classify2.weight)		

	def forward(self, left, right, topo_fea, loss_fun2, loss_fun1):

		# code position feature
		ass_mat= torch.cat((torch.cat((self.features_list[2], self.features_list[-1]), dim= 1), torch.cat((self.features_list[-1].T, self.features_list[3]), dim= 1)), dim= 0).to(torch.float)
		ass_mat2= torch.cat((torch.cat((self.features_list[0], self.features_list[-1]), dim= 1), torch.cat((self.features_list[-1].T, self.features_list[1]), dim= 1)), dim= 0).to(torch.float)
		sloc= self.encoder4sloc(ass_mat)
		sloc0= sloc.clone()
		# encode feature for Net1
		drug_emb, micro_emb= self.encoder4drug(self.features_list[0]), self.encoder4micro(self.features_list[1])		
		h1= torch.cat((drug_emb, micro_emb), dim= 0)
		# encode feature for Net2
		drug_emb2, micro_emb2= self.encoder4drug2(ass_mat2[0: 1373]), self.encoder4micro2(ass_mat2[1373:])
		h2= torch.cat((drug_emb2, micro_emb2), dim= 0)
		emb1, emb2= [], []
		# transformer
		emb2.append(torch.cat((ass_mat2, h2), dim= 1))
		for tf_layer in self.tf_layers:
			h2= tf_layer(h2, self.edge_type_emb)
			emb2.append(h2)
		# GAT, (64+ 64+ 3)* 3
		emb1.append(torch.cat((ass_mat, h1, sloc, topo_fea), dim= 1))
		for l in range(self.num_layers):
			h1, sloc, topo_fea= self.gat_layers[l](self.g, h1, sloc, topo_fea)
			emb1.append(torch.cat((h1, sloc, topo_fea), dim= 1))
		emb1, emb2= torch.cat(emb1, dim= 1), torch.cat(emb2, dim= 1)
		batch_label0, right4micro= torch.zeros(1546).to(torch.long).cuda(), right- 1373
		# loss
		loss4gat_recon= loss_fun2(self.decoder4drug(drug_emb[left]), self.features_list[0][left])+ loss_fun2(self.decoder4micro(micro_emb[right4micro]), self.features_list[1][right4micro]) + loss_fun2(self.decoder4sloc(sloc0[left]), ass_mat[left])+ loss_fun2(self.decoder4sloc(sloc0[right]), ass_mat[right])
		loss4gat_classify= loss_fun1(self.decoder4classify(drug_emb[left]), batch_label0[left])+ loss_fun1(self.decoder4classify(micro_emb[right4micro]), batch_label0[right4micro]+ 1)
		loss4transformer_recon= loss_fun2(self.decoder4drug2(drug_emb2[left]), ass_mat2[left])+ loss_fun2(self.decoder4micro2(micro_emb2[right4micro]), ass_mat2[right])
		loss4transformer_classify= loss_fun1(self.decoder4classify2(drug_emb2[left]), batch_label0[left])+ loss_fun1(self.decoder4classify2(micro_emb2[right4micro]), batch_label0[right]+ 1)
		return loss4gat_recon+ loss4gat_classify+ loss4transformer_recon+ loss4transformer_classify, self.decoder(emb1[left], emb1[right], emb2[left], emb2[right])
