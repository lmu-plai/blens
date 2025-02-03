"""
This file is part of the BLens binary function captioner.
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import functools

import torch
import torch.nn as nn

from .layers.decoder import TransformerDecoderTextualHead, CaptioningModel


class LORD(nn.Module):
	def __init__(
		self,
		tokenizer,
		weights=None,
		params=None
	):
		super().__init__()		
		
				
		text_decoder = TransformerDecoderTextualHead(
			visual_feature_size=params['global']['visual_feature_size'],
			vocab_size=tokenizer.vocabulary,
			hidden_size=params['global']['visual_feature_size'],
			num_layers=params['LORD']['LORD']['depth'],
			attention_heads=params['LORD']['LORD']['head'],
			feedforward_size=params['LORD']['LORD']['feedforward_factor']*params['global']['visual_feature_size'],
			max_caption_length=params['global']['max_tokens'] + 2,
   			coca_dim=params['global']['visual_feature_size'],
			coca_num_tokens=tokenizer.vocabulary,
			coca_unimodal_depth=params['COMBO']['CoCa']["multimodal_depth"],
			coca_dim_head=params['COMBO']['CoCa']["dim_head"],
			coca_heads=params['COMBO']['CoCa']["head"],
			coca_ff_mult=params['COMBO']['CoCa']['feedforward_factor'],
			coca_weights=weights,
			mask_future_positions=True,
			padding_idx=tokenizer.pad_token_id,
			decoder_type='bert_en',
			visual_projection_type='linearLn',
			norm_type="pre",
			dropout=params['LORD']['LORD']['dropout'],
			final_mlp=params['LORD']['LORD']['final_mlp_depth'],
		)


		self.captioner = CaptioningModel(
			text_decoder,
			sos_index=tokenizer.sos_token_id,
			eos_index=tokenizer.eos_token_id,
			loss_type=params['LORD']['LORD']['loss_type'],
			max_steps=params['global']['max_tokens'] + 2,
			decoder_type=params['global']['decoder_type'],
			vocab_size=tokenizer.vocabulary
		)

	def forward(self, batch):
		return self.captioner(batch)
