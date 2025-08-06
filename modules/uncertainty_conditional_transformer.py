import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear

class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pe", pe)

	def forward(self, x):
		x = x + self.pe[: x.size(0), :]
		return self.dropout(x)


class ConcatSquashLinear(Module):
	def __init__(self, dim_in, dim_out, dim_ctx):
		super(ConcatSquashLinear, self).__init__()
		self._layer = Linear(dim_in, dim_out)
		self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
		self._hyper_gate = Linear(dim_ctx, dim_out)

	def forward(self, ctx, x):
		# ctx: (B, 1, F+3)
		# x: (B, T, 2)
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		# if x.dim() == 3:
		#     gate = gate.unsqueeze(1)
		#     bias = bias.unsqueeze(1)
		ret = self._layer(x) * gate + bias
		return ret
	
	def batch_generate(self, ctx, x):
		# ctx: (B, n, 1, F+3)
		# x: (B, n, T, 2)
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		# if x.dim() == 3:
		#     gate = gate.unsqueeze(1)
		#     bias = bias.unsqueeze(1)
		ret = self._layer(x) * gate + bias
		return ret
	

class st_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 2
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 256
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X):
		'''
		X: b, T, 2

		return: b, F
		'''
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)
		state_x = state_x.squeeze(0)

		return state_x


class social_transformer(nn.Module):
	def __init__(self):
		super(social_transformer, self).__init__()
		self.encode_past = nn.Linear(60, 256, bias=False)
		# self.encode_past = nn.Linear(48, 256, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		# print(h_feat.shape)
		# n_samples, 1, 64
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_

		return h_feat


class TransformerDenoisingModel(Module):

	def __init__(self, context_dim=256, tf_layer=2):
		super().__init__()
		self.encoder_context = social_transformer()
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
		self.concat1 = ConcatSquashLinear(2, 2*context_dim, context_dim+3)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
		self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)


	def forward(self, x, beta, context, mask):
		batch_size = x.size(0)
		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)


		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)


		ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
		
		x = self.concat1(ctx_emb, x)
		final_emb = x.permute(1,0,2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1,0,2)
		trans = self.concat3(ctx_emb, trans)
		trans = self.concat4(ctx_emb, trans)
		return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, x, beta, context, mask):
		batch_size = x.size(0)
		beta = beta.view(beta.size(0), 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		# time_emb: [11, 1, 3]
		# context: [11, 1, 256]
		ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, 10, 1).unsqueeze(2)
		# x: 11, 10, 20, 2
		# ctx_emb: 11, 10, 1, 259
		x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, 20, 512)
		# x: 110, 20, 512
		final_emb = x.permute(1, 0, 2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 10, 20, 512)
		# trans: 11, 10, 20, 512
		trans = self.concat3.batch_generate(ctx_emb, trans)
		trans = self.concat4.batch_generate(ctx_emb, trans)
		return self.linear.batch_generate(ctx_emb, trans)
	


if __name__ == '__main__':

	# x = torch.rand(2, 6, 2) # batch_size, t, 2
    # # noise = torch.randn(2, 6, 2)
    # # global_feature = torch.randn(2, 256)
    # # timestep = torch.randint(0, 1000, (2,)) 
	# model = TransformerDenoisingModel()
    # # noise_pred = noise_pred_net(noise, timestep=timestep, global_cond=global_feature)

	# print(model)

	# 1. time
	# timesteps = timestep
	# if not torch.is_tensor(timesteps):
	# 	# TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
	# 	timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
	# elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
	# 	timesteps = timesteps[None].to(sample.device)
	# # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
	# timesteps = timesteps.expand(sample.shape[0])

	   

	r_pt2pl = torch.randn(7729, 3)
	# r_pt2pl = torch.randn(7730, 3)
	r_pt2pl = r_pt2pl.unsqueeze(0)
	print(r_pt2pl.shape)

	padding_flag = False
	if r_pt2pl.shape[1] % 2 != 0:
		# 如果是奇数，在第二个维度末尾补零
		padding = torch.zeros(r_pt2pl.shape[0], 1, r_pt2pl.shape[2])
		r_pt2pl = torch.cat((r_pt2pl, padding), dim=1)
		padding_flag = True

	print(r_pt2pl.shape)


	local_cond = r_pt2pl
	noise = torch.randn_like(local_cond)
	noise = noise.to(local_cond.device)
	timestep = torch.randint(0, 1000, (1,))
	timestep = timestep.to(local_cond.device)

	print(noise)
	print(timestep)

    # noise_pred = noise_pred_net(noise, timestep=timestep, local_cond=local_cond, global_cond=None)

    # if padding_flag:
    #     noise_pred = noise_pred[:, :-1, :]  # 去除最后一行
    
    # r_pt2pl = noise_pred.squeeze(0)


    # print(r_pt2pl.shape)