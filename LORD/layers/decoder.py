import warnings
import functools
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from COMBO.coca.coca_pytorch import Residual, ParallelTransformerBlock, LayerNorm
from .bert import BertConfig
from .bert.modeling_bert import BertEncoder

class TextualHead(nn.Module):
	def __init__(self,
				 visual_feature_size: int, vocab_size: int, hidden_size: int):
		super().__init__()
		self.visual_feature_size = visual_feature_size
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size

	@property
	def textual_feature_size(self):
		return self.hidden_size

def create_projecton_layer(visual_projection_type,
						   visual_feature_size,
						   textual_feature_size,
						   ):
	if visual_projection_type is None:
		visual_projection = nn.Linear(
			visual_feature_size, textual_feature_size
		)
	elif visual_projection_type == 'linearLn':
		visual_projection = nn.Sequential(
			nn.Linear(
				visual_feature_size, textual_feature_size
			),
			nn.LayerNorm(textual_feature_size),
		)
	else:
		raise NotImplementedError(visual_projection_type)
	return visual_projection

class WordAndPositionalEmbedding(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		hidden_size: int,
		dropout: float = 0.1,
		max_caption_length: int = 30,
		padding_idx: int = 0,
	):
		super().__init__()
		self.vocab_size = vocab_size
		#self.padding_idx = padding_idx

		#self.words = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
		self.words = nn.Embedding(vocab_size, hidden_size)

		# We provide no "padding index" for positional embeddings. We zero out
		# the positional embeddings of padded positions as a post-processing.
		self.positions = nn.Embedding(max_caption_length, hidden_size)
		self.layer_norm = nn.LayerNorm(
			hidden_size, eps=1e-8, elementwise_affine=True
		)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, tokens):
		batch_size, caption_length = tokens.size()
		position_indices = self._create_position_indices(batch_size, caption_length, tokens.device)

		# shape: (batch_size, max_caption_length, hidden_size)
		word_embeddings = self.words(tokens)
		position_embeddings = self.positions(position_indices)

		# shape: (batch_size, max_caption_length, hidden_size)
		embeddings = self.layer_norm(word_embeddings + position_embeddings)
		embeddings = self.dropout(embeddings)

		#token_mask = (tokens != self.padding_idx).unsqueeze(-1)
		#embeddings = embeddings * token_mask.type(embeddings.dtype)
		return embeddings

	@functools.lru_cache(maxsize=128)
	def _create_position_indices(self, batch_size, caption_length, device):
		positions = torch.arange(caption_length, dtype=torch.int32, device=device)
		positions = positions.unsqueeze(0).expand(batch_size, caption_length)
		return positions

class BertEncoderAsDecoder(nn.Module):
	def __init__(self, encoder):
		super().__init__()
		self.encoder = encoder

	def forward(self, tgt, memory,
				tgt_mask=None,
				#memory_mask=None,
				tgt_key_padding_mask=None,
				memory_key_padding_mask=None,
				tgt_bi_valid_mask=None,
				encoder_history_states=None,
				# tgt_bi_valid_mask: N x num_tgt
				):
		assert tgt_key_padding_mask is None, 'not supported'
		assert tgt_mask.dim() == 2
		assert tgt_mask.shape[0] == tgt_mask.shape[1]
		# tgt_mask should always be 0/negative infinity
		# mask
		tgt = tgt.transpose(0, 1)
		memory = memory.transpose(0, 1)

		hidden_states = torch.cat((memory, tgt), dim=1)
		num_tgt = tgt.shape[1]
		num_memory = memory.shape[1]
		device = tgt.device
		dtype = tgt.dtype
		top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
		top_right = torch.full((num_memory, num_tgt), float('-inf'), device=tgt.device, dtype=dtype,)
		bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device,)
		left = torch.cat((top_left, bottom_left), dim=0)
		right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

		full_attention_mask = torch.cat((left, right), dim=1)[None, :]

		if memory_key_padding_mask is None:
			memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
		# if it is False, it means valid. That is, it is not a padding
		assert memory_key_padding_mask.dtype == torch.bool
		zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
		zero_negative_infinity[memory_key_padding_mask] = float('-inf')
		full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + num_tgt))
		full_attention_mask = full_attention_mask.clone()
		origin_left = full_attention_mask[:, :, :num_memory]
		update = zero_negative_infinity[:, None, :]
		full_attention_mask[:, :, :num_memory] = origin_left + update

		if tgt_bi_valid_mask is not None:
			# verify the correctness
			bs = full_attention_mask.shape[0]
			# during inference, tgt_bi_valid_mask's length is not changed, but
			# num_tgt can be increased
			max_valid_target = tgt_bi_valid_mask.shape[1]
			mask = tgt_bi_valid_mask[:, None, :].expand((bs, num_memory+num_tgt, max_valid_target))
			full_attention_mask[:, :, num_memory:(num_memory+max_valid_target)][mask] = 0

		# add axis for multi-head
		full_attention_mask = full_attention_mask[:, None, :, :]

		if encoder_history_states is None:
			result = self.encoder(
				hidden_states=hidden_states,
				attention_mask=full_attention_mask,
				encoder_history_states=encoder_history_states,
			)
			result = list(result)
			result[0] = result[0][:, num_memory:].transpose(0, 1)
			if self.encoder.output_hidden_states:
				return result[0], result[1]
			else:
				# make it back-compatible
				return result[0]
		else:
			encoder_out = self.encoder(
				hidden_states=hidden_states[:, -1:],
				attention_mask=full_attention_mask[:, :, -1:],
				encoder_history_states=encoder_history_states,
			)
			result = encoder_out[0].transpose(0, 1)
			if self.encoder.output_hidden_states:
				return result, encoder_out[1]
			else:
				return result

def create_decoder(vocab_size, decoder_type, norm_type,
				   textual_feature_size,
				   attention_heads,
				   feedforward_size,
				   dropout,
				   num_layers,
				   output_hidden_states=False,
				   use_mlp_wrapper=None,
				   ):
	assert norm_type in ['post', 'pre']

	config = BertConfig(
		vocab_size_or_config_json_file=vocab_size,
		hidden_size=textual_feature_size,
		num_hidden_layers=num_layers,
		num_attention_heads=attention_heads,
		intermediate_size=feedforward_size,
		hidden_act="gelu",
		hidden_dropout_prob=dropout,
		attention_probs_dropout_prob=dropout,
		layer_norm_eps=1e-12,
	)
	config.pre_norm=(norm_type == 'pre')
	config.use_mlp_wrapper = use_mlp_wrapper
	config.output_hidden_states = output_hidden_states
	encoder = BertEncoder(config)
	return BertEncoderAsDecoder(encoder)


class TransformerDecoderTextualHead(TextualHead):
	# used by unifusiondecoder and imageencodertextdecoder pipelines
	def __init__(
		self,
		visual_feature_size: int,
		vocab_size: int,
		hidden_size: int,
		num_layers: int,
		attention_heads: int,
		feedforward_size: int,
        coca_dim,
		coca_num_tokens,
		coca_unimodal_depth,
		coca_dim_head=64,
		coca_heads=8,
		coca_ff_mult=4,
		coca_weights=None,
		dropout: float = 0.1,
		norm_type: str = "post",
		mask_future_positions: bool = True,
		max_caption_length: int = 30,
		padding_idx: int = 0,
		decoder_type=None,
		visual_projection_type=None,
		not_tie_weight=None,
		output_hidden_states=None,
		use_mlp_wrapper=None,
		cosine_linear=False,
        final_mlp=1,

	):
		super().__init__(visual_feature_size, vocab_size, hidden_size)
		self.num_layers = num_layers
		self.attention_heads = attention_heads
		self.feedforward_size = feedforward_size
		self.dropout = dropout
		assert mask_future_positions
		self.padding_idx = padding_idx


		self.embedding = CoCaTextEmbedding(
			dim=coca_dim,
			num_tokens=coca_num_tokens,
			unimodal_depth=coca_unimodal_depth,
			dim_head=coca_dim_head,
			heads=coca_heads,
			ff_mult=coca_ff_mult,
			pad_id=padding_idx,
			dropout=dropout,
			max_caption_length=max_caption_length,
			weights=coca_weights
		)


		if visual_feature_size is not None:
			self.visual_projection = create_projecton_layer(
				visual_projection_type, visual_feature_size, self.textual_feature_size)
		else:
			self.visual_projection = nn.Identity()

		self.transformer = create_decoder(
			vocab_size=vocab_size,
			decoder_type=decoder_type,
			norm_type=norm_type,
			textual_feature_size=self.textual_feature_size,
			attention_heads=self.attention_heads,
			feedforward_size=self.feedforward_size,
			dropout=dropout,
			num_layers=self.num_layers,
			output_hidden_states=output_hidden_states,
			use_mlp_wrapper=use_mlp_wrapper,
		)
		self.apply(self._init_weights)

		# Create an output linear layer and tie the input and output word
		# embeddings to reduce parametejs.
		self.output = nn.Linear(self.textual_feature_size, vocab_size)
		if not not_tie_weight:			
			self.output.weight = self.embedding.token_emb.weight


	@staticmethod
	def _init_weights(module):
		r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""
		if isinstance(module, nn.Linear):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.MultiheadAttention):
			module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
			module.out_proj.weight.data.normal_(mean=0.0, std=0.02)


	def forward(
		self,
		hidden_states,
		caption_tokens,
		hidden_valid_mask=None, # can be None
		caption_lengths=None, # useless
		bi_valid_mask_caption=None,
		#caption_mask=None,
		encoder_history_states=None,
		return_dict=False,
		multi_decoding=False
	):

		if return_dict:
			ret = {}

		projected_visual_features = self.visual_projection(hidden_states) if hidden_states is not None else None
		if return_dict:
			ret['projected_visual_features'] = projected_visual_features
		batch_size, max_caption_length = caption_tokens.size()
		caption_embeddings = self.embedding(caption_tokens)

		# An additive mask for masking the future (one direction).
		uni_mask_zero_neg = self._generate_future_mask(
			max_caption_length, caption_embeddings.dtype, caption_embeddings.device, multi_decoding
		)

		# We transpose the first two dimensions of tokens embeddings and visual
		# features, as required by decoder.
		caption_embeddings = caption_embeddings.transpose(0, 1)
		if projected_visual_features is not None:
			projected_visual_features = projected_visual_features.transpose(0, 1)
		else:
			projected_visual_features = torch.zeros(
				(0, caption_embeddings.shape[1], caption_embeddings.shape[2]),
				dtype=caption_embeddings.dtype,
				device=caption_embeddings.device,
			)
		extra_param = {}

		if bi_valid_mask_caption is not None:
			extra_param = {'tgt_bi_valid_mask': bi_valid_mask_caption}
		if not isinstance(self.transformer, torch.nn.modules.transformer.TransformerDecoder):
			extra_param['encoder_history_states'] = encoder_history_states

		# if transformer here is the pytorch/decoder, there is no chance, the
		# output is always tensor
		trans_out = self.transformer(
			caption_embeddings,
			projected_visual_features,
			memory_key_padding_mask=(hidden_valid_mask.logical_not() if hidden_valid_mask is not None else None),
			tgt_mask=uni_mask_zero_neg,
			#tgt_key_padding_mask=caption_mask,
			#encoder_history_states=encoder_history_states,
			**extra_param,
		)
		if isinstance(trans_out, tuple):
			textual_features = trans_out[0]
		else:
			assert isinstance(trans_out, torch.Tensor)
			textual_features = trans_out
		# Undo the transpose and bring batch to dim 0.
		# shape: (batch_size, max_caption_length, hidden_size)
		textual_features = textual_features.transpose(0, 1)
		if return_dict:
			ret['textual_features'] = textual_features

		# shape: (batch_size, max_caption_length, vocab_size)
		output_logits = self.output(textual_features)

		if isinstance(trans_out, tuple):
			if return_dict:
				ret['output_logits'] = output_logits
				ret['history'] = trans_out[1]
				return ret
			else:
				return output_logits, trans_out[1]
		else:
			if return_dict:
				ret['output_logits'] = output_logits
				return ret
			else:
				return output_logits

	def _generate_future_mask(
		self, size: int, dtype: torch.dtype, device: torch.device, multi_decoding
	) -> torch.Tensor:
		# Default mask is for forward direction. Flip for backward direction.
		mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
		if multi_decoding:
			return mask.masked_fill(mask == 1, 0)
		return mask.masked_fill(mask == 1, float("-inf"))

def convert2valid(shape, length=None, device='cuda'):
	if length is None:
		valid = torch.full(shape, fill_value=True, device=device)
	else:
		ones = torch.ones(shape, device=device)
		valid = ones.cumsum(dim=1) <= length.unsqueeze(1)
	return valid

class SmoothLabelCrossEntropyLoss(nn.Module):
	def __init__(self, eps=0.1, log_prefix='', ignore_index=None):
		super().__init__()
		self.eps = eps
		self.log_soft = nn.LogSoftmax(dim=1)
		self.kl = nn.KLDivLoss(reduction='none')

		self.iter = 0
		self.max_loss = 0
		self.min_loss = 0
		self.log_prefix = log_prefix
		self.ignore_index = ignore_index

	def forward(self, feature, target):
		# if it is fp16, convert it to fp32 explicitly as some trainer will not do automatically
		feature = feature.float()

		if self.ignore_index is not None:
			valid_mask = target != self.ignore_index
			target = target[valid_mask]
			feature = feature[valid_mask]
		assert target.numel() > 0
		debug_print = (self.iter % 100) == 0
		self.iter += 1
		eps = self.eps
		n_class = feature.size(1)
		one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
		one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
		log_prb = self.log_soft(feature)
		loss = self.kl(log_prb, one_hot)
		return loss.sum(dim=1).mean()

class CaptioningModel(nn.Module):
	def __init__(
		self,
		textual,
		sos_index=1,
		eos_index=2,
		loss_type=None,
		max_steps=None,
		decoder_type=None,
		vocab_size=None		
	):
		super().__init__()
		self.textual = textual
		
		self.padding_idx = self.textual.padding_idx
		self.sos_index = sos_index
		self.eos_index = eos_index
		self.max_steps = max_steps
		self.decoder_type = decoder_type
		self.vocab_size = vocab_size
		

		if loss_type == "hard":
			self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
		elif loss_type == 'smooth':
			self.loss = SmoothLabelCrossEntropyLoss(ignore_index=self.padding_idx)
		else:
			raise NotImplementedError(loss_type)

	def forward(self, batch):
		result = self.forward_one(batch)
		return result

	def forward_one(self, batch):
		# shape: (batch_size, max_caption_length, vocab_size)
		if 'image_tokens' in batch:
			visual_features = batch['image_tokens']
		else:
			visual_features = None
		visual_features_valid = None
		return self.forward_one_ce(batch, visual_features, visual_features_valid)

	def forward_one_ce(self, batch, visual_features, visual_features_valid):
		has_image = (visual_features is not None)
		assert has_image == ('image_tokens' in batch)
		
		if self.training:

			if self.decoder_type in ['lord']:
				# Randomly mask tokens with sos + mask eos and pad tokens
				caption_token_input = batch["caption_tokens"].clone()
				caption_token_input[caption_token_input == 0] = 1
				caption_token_input[caption_token_input == 2] = 1


				for b in range(caption_token_input.shape[0]):
					valuesPossiblyMasked = caption_token_input[b] > 2
					valuesPossiblyMasked = torch.flatten(valuesPossiblyMasked.nonzero()).cpu().numpy()
					n = len(valuesPossiblyMasked)
					if n == 0:
						continue

					probs = torch.nn.functional.softmax(torch.linspace(1, 2, steps=(n+1)), dim=-1).numpy()
					numberOfMasked = np.random.choice([i for i in range(n+1)], p=probs)
					maskedSubSet = np.random.choice(valuesPossiblyMasked, numberOfMasked, replace=False)
					maskedSubSet = torch.tensor(maskedSubSet, device=caption_token_input.device)
					caption_token_input[b, maskedSubSet] = 1 # randomly masked (mimic a token not inferred yet)
			else:
				caption_token_input = batch["caption_tokens"]


			output_logits = self.textual(
				visual_features,
				caption_token_input,
				hidden_valid_mask=visual_features_valid,
				bi_valid_mask_caption=batch.get('bi_valid_mask_caption'),
				multi_decoding=self.decoder_type in ['lord']
			)

			if 'need_predict' in batch:
				target = batch["caption_tokens"].clone()
				if self.padding_idx is not None:
					target[batch['need_predict'] == 0] = self.padding_idx
			else:
				assert ValueError()

			need_predict = batch['need_predict']
			feat = output_logits[:, :-1].contiguous()
			target = target[:, 1:].contiguous()
			need_predict = need_predict[:, 1:].contiguous()
			feat = feat.reshape(-1, self.textual.vocab_size)
			target = target.view(-1)
			need_predict = need_predict.view(-1)

			valid_mask = need_predict == 1

			target = target[valid_mask]
			feat = feat[valid_mask]
			return  self.loss(feat, target)

		else:

			if self.decoder_type ==  'lord':
				output_dict = self.lord_decoding(batch, visual_features, visual_features_valid)
			else:
				output_dict = self.simple_decoding(batch, visual_features, visual_features_valid)

		return output_dict


	def simple_decoding(self, batch, visual_features, visual_features_valid):
		bi_valid_mask_caption =  batch.get('bi_valid_mask_caption')

		caption_lengths = self.max_steps
		batch_size = visual_features.size(0)

		tokens = visual_features.new_full( (batch_size,1), 1).long()
		scoresB = []

		for j in range(1, caption_lengths):

			scores = self.textual(
				visual_features,
				tokens,
				caption_lengths=caption_lengths,
				hidden_valid_mask=visual_features_valid,
				bi_valid_mask_caption=bi_valid_mask_caption,
				encoder_history_states=None,
				multi_decoding=False
			)

			scores = F.softmax(scores, dim=-1)  # (batch_size, j+1, vocab_size)
			scores = scores[:, -1, :]
			
			# Do not predict SOS and PAD labels
			scores[:, 0:2] = 0

			# Save scores
			scoresB += [ scores ] # (batch_size, vocab_size)

			next_token = torch.argmax(scores, dim=-1, keepdim=True) # (batch_size, 1)
			tokens = torch.cat((tokens, next_token), dim=1) # (batch_size, j+2)
			

		scoresB = torch.stack(scoresB, dim=-1) # (batch_size, vocab_size, caption_lengths - 1)
		scoresB = torch.swapaxes(scoresB, 1, 2) # (batch_size, caption_lengths - 1, vocab_size)

		output_dict = {
			'predictions': tokens,		# naive predictions without ending the sentence
			'scores': scoresB,		# actual scores for each token at each position
		}

		return output_dict

	def lord_decoding(self, batch, visual_features, visual_features_valid):		
		bi_valid_mask_caption =  batch.get('bi_valid_mask_caption')

		batch_size = visual_features.size(0)
		caption_lengths = self.max_steps
		vocab_size = self.vocab_size

		tokens = visual_features.new_full( (batch_size, caption_lengths), self.sos_index).long()

		prediciton_mask = visual_features.new_full( (batch_size, caption_lengths, vocab_size), 0)
		prediciton_mask[:, -1, :] = 1 # token predicted after eos

		predictionsScores = [ [] for b in range(batch_size)]

		for j in range(1, caption_lengths):

			scores = self.textual(
				visual_features,
				tokens,
				caption_lengths=caption_lengths,
				hidden_valid_mask=visual_features_valid,
				bi_valid_mask_caption=bi_valid_mask_caption,
				encoder_history_states=None,
				multi_decoding=True
			)


			scores = F.softmax(scores, dim=-1)  # (batch_size, caption_lengths, vocab_size

			# Do not predict SOS and PAD labels
			scores[:, :, 0:2] = 0

			# Do not predict already predicted tokens
			scores[prediciton_mask == 1] = 0

			max_values, max_indices = torch.max(scores.view(batch_size, -1), dim=1)
			cooToken = max_indices // vocab_size
			cooTokenShft = cooToken + 1
			cooTokenShft = torch.clamp(cooTokenShft, max=caption_lengths-1) # predicted token 1 is at 0
			cooVoc =  max_indices % vocab_size

			tokens[torch.arange(batch_size), cooTokenShft] = cooVoc # Select the most probable token
			prediciton_mask[torch.arange(batch_size), cooToken, :] = 1 # This token is predicted

			# Special labels are never seen during predictions
			tokens[tokens < 3] = self.sos_index

			# Save 'confidence' for threshold optimization later
			for b in range(batch_size):
				predictionsScores[b] += [(max_values[b].cpu().item(),cooTokenShft[b].cpu().item(),cooVoc[b].cpu().item())]

		output_dict = {
			'predictions': tokens,			# naive predictions without stopping
			'scores': predictionsScores,	# 'confidence' for threshold optimization later
		}

		return output_dict

class CoCaTextEmbedding(nn.Embedding):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        unimodal_depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        pad_id=0,
        dropout=0.2,
        max_caption_length=30,
        weights=None
    ):
        super().__init__(num_tokens, dim)
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.text_cls_token = nn.Parameter(torch.randn(dim))
        self.pad_id = pad_id
        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult), dropout=dropout),
            )
        self.positions = nn.Embedding(max_caption_length, dim)
        self.layer_norm = nn.LayerNorm(
			dim, eps=1e-8, elementwise_affine=True
		)
        if(weights):
            with torch.no_grad():
                self.token_emb.weight.copy_(weights["coca.token_emb.weight"])
                self.text_cls_token.copy_(weights["coca.text_cls_token"])
                for ind in range(unimodal_depth):
                    self.unimodal_layers[ind].fn.norm.gamma.copy_(weights[f"coca.unimodal_layers.{ind}.fn.norm.gamma"])
                    self.unimodal_layers[ind].fn.norm.beta.copy_(weights[f"coca.unimodal_layers.{ind}.fn.norm.beta"])
                    self.unimodal_layers[ind].fn.rotary_emb.inv_freq.copy_(weights[f"coca.unimodal_layers.{ind}.fn.rotary_emb.inv_freq"])
                    self.unimodal_layers[ind].fn.fused_attn_ff_proj.weight.copy_(weights[f"coca.unimodal_layers.{ind}.fn.fused_attn_ff_proj.weight"])
                    self.unimodal_layers[ind].fn.attn_out.weight.copy_(weights[f"coca.unimodal_layers.{ind}.fn.attn_out.weight"])
                    self.unimodal_layers[ind].fn.ff_out[1].weight.copy_(weights[f"coca.unimodal_layers.{ind}.fn.ff_out.1.weight"])


    def forward(self, text):
        batch, device = text.shape[0], text.device
        seq = text.shape[1]
        position_indices = self._create_position_indices(batch, seq, device)
		# shape: (batch_size, max_caption_length, hidden_size)
        position_embeddings = self.positions(position_indices)

        text_tokens = self.token_emb(text)

        # append text cls tokens
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

        # create specific mask for text cls token at the end
        # to prevent it from attending to padding

        cls_mask = rearrange(text!=self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # go through unimodal layers

        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)

        # get text cls token

        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]

        return self.layer_norm(text_tokens+position_embeddings)

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, batch_size, caption_length, device):
        positions = torch.arange(caption_length, dtype=torch.int32, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, caption_length)
        return positions
