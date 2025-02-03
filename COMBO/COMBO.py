"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import torch
import torch.nn as nn
from torch.nn import functional as F

from .coca.coca_pytorch import CoCa
from .Dexter2Seq import Dexter2Seq
from .PalmTreeSeq import PalmTreeSeq

class COMBO(nn.Module):
	def __init__(
		self,
		tokenizer,
		params
	):
		super().__init__()

		self.dexter = params["global"]["dexter"]
		self.clap = params["global"]["clap"]
		self.palmtree = params["global"]["palmtree"]

		self.no_combo = params["global"]["no_combo"]


		if self.dexter:
			self.dexterSeq = Dexter2Seq(patches=params['COMBO']['Dexter2Seq']["patches"], dim=512,  visual_feature_size=params['global']['visual_feature_size'], dimInter=params['COMBO']['Dexter2Seq']["dim_inter"])

		if self.clap:
			self.clapSeq = Dexter2Seq(patches=params['COMBO']['Clap2Seq']["patches"], dim=768,  visual_feature_size=params['global']['visual_feature_size'], dimInter=params['COMBO']['Clap2Seq']["dim_inter"])

		if self.palmtree:
			self.palmtreeSeq = PalmTreeSeq(size=params['COMBO']['PalmtreeSeq']['size'], visual_feature_size=params['global']['visual_feature_size'], palmtreeemb_size=params['COMBO']['PalmtreeSeq']['emb_size'])


		self.coca = CoCa(
			dim = params['global']['visual_feature_size'],                         		# model dimension
			img_encoder = None,                						# vision transformer - image encoder, returning image embeddings as (batch, seq, dim)
			image_dim = params['global']["visual_feature_size"],                    	# image embedding dimension, if not the same as model dimensions
			num_tokens = tokenizer.vocabulary,  						# number of text tokens
			unimodal_depth = params['COMBO']['CoCa']["unimodal_depth"],                 	# depth of the unimodal transformer
			multimodal_depth = params['COMBO']['CoCa']["multimodal_depth"],              	# depth of the multimodal transformer
			dim_head = params['COMBO']['CoCa']["dim_head"],                      		# dimension per attention head
			heads = params['COMBO']['CoCa']["head"],                      			# number of attention heads
			caption_loss_weight = params['COMBO']['CoCa']["caption_loss_weight"],        	# weight on the autoregressive caption loss
			contrastive_loss_weight = params['COMBO']['CoCa']["contrastive_loss_weight"],	# weight on the contrastive loss between image and text CLS embeddings
			pad_id = tokenizer.pad_token_id,
			num_img_queries=params['COMBO']['CoCa']["num_img_queries"],			# lenght of final sequence ( + 1)
			ff_mult=params['COMBO']['CoCa']['feedforward_factor'],				# factor in the inner dimension of the feed forward
			dropout=params['COMBO']['CoCa']['dropout'],					# dropout inside CoCa residuals
			final_mlp=params['COMBO']['CoCa']['final_mlp_depth'],				# depth of the the final MLP to predict tokens
			)


	def forward(self, batch):
		loss = self.coca(text = batch["caption_tokens"], image_tokens = self.combo_input(batch), return_loss = True)
		return loss

	def image_embedding(self, batch):
		combo_input = self.combo_input(batch)
		if self.no_combo:
			return combo_input
		return self.coca.embed_image(image_tokens = combo_input)

	def combo_input(self, batch):
		seqL = []

		if self.dexter:
			seqL += [self.dexterSeq(batch["dexter"])]

		if self.clap:
			seqL += [self.clapSeq(batch["clap"])]

		if self.palmtree:
			seqL += [self.palmtreeSeq(batch["palmtree"])]

		tokens = torch.cat(seqL, 1)

		return tokens
		
	def ensembleEncoderParameters(self):
		parameters = {}
		
		if self.dexter:
			parameters["DEXTER"] = sum(p.numel() for p in self.dexterSeq.parameters() if p.requires_grad)

		if self.clap:
			parameters["CLAP"] = sum(p.numel() for p in self.clapSeq.parameters() if p.requires_grad)

		if self.palmtree:
			parameters["PalmTree"] = sum(p.numel() for p in self.palmtreeSeq.parameters() if p.requires_grad)

		print(parameters)
		return sum([parameters[x] for x in parameters])

	def functionEncoderParameters(self):
		parameters = {}

		parameters["queries"] = self.coca.img_queries.numel()
		parameters["cross attention"] = sum(p.numel() for p in self.coca.img_attn_pool.parameters() if p.requires_grad)
		parameters["layer normalization"] = sum(p.numel() for p in self.coca.img_attn_pool_norm.parameters() if p.requires_grad)

		print(parameters)
		return sum([parameters[x] for x in parameters])

	def decode(self, batch, decoder_type):
		if decoder_type == 'simple':
			return self.simple_decoding(batch)
		assert(False)

	def simple_decoding(self, batch):
		image_tokens = self.combo_input(batch)

		caption_lengths = 20
		batch_size = image_tokens.size(0)
		text = image_tokens.new_full( (batch_size,1), 1).long() # (batch_size, 1)
		scoresB = []

		for j in range(1, caption_lengths):
			scores = self.coca(text=text, image_tokens=image_tokens, return_loss = False)
			#print(scores.shape)
			scores = F.softmax(scores, dim=-1)  # (batch_size, j, vocab_size)
			#print(scores.shape)
			scores = scores[:, -1, :]

			# Do not predict SOS and PAD labels
			scores[:, 0:2] = 0

			# Save scores
			scoresB += [ scores ] # (batch_size, vocab_size)

			next_token = torch.argmax(scores, dim=-1, keepdim=True) # (batch_size, 1)
			text = torch.cat((text, next_token), dim=1) # (batch_size, j+1)


		scoresB = torch.stack(scoresB, dim=-1) # (batch_size, vocab_size, caption_lengths - 1)
		scoresB = torch.swapaxes(scoresB, 1, 2) # (batch_size, caption_lengths - 1, vocab_size)

		output_dict = {
			'predictions': text,		# naive predictions without ending the sentence
			'scores': scoresB,		# actual scores for each token at each position
		}

		return output_dict
