"""
This file is part of the BLens binary function captioner.
The 'recursive_to_device' and 'collate_fn' functions are components from the official GIT implementation by Microsoft Corporation, licensed under the MIT License (see LORD/layers/LICENSE).
 
Copyright (C) 2024-2025 by Tristan Benoit, Yunru Wang, Moritz Dannehl, and Johannes Kinder.
Copyright (c) Microsoft Corporation.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""



import pickle
import hashlib
from tqdm import tqdm

from torch.utils.data.dataloader import default_collate
import torch
import numpy as np

def loadNLPData(path):
	with open(path, "rb") as f:
		data = pickle.load(f)  # [0,1,2] -> [binPath, vaddr, real_name, name, vaddr, bId, fId]
	return data

def recursive_to_device(d, device, **kwargs):
	if isinstance(d, tuple) or isinstance(d, list):
		return [recursive_to_device(x, device, **kwargs) for x in d]
	elif isinstance(d, dict):
		return dict((k, recursive_to_device(v, device)) for k, v in d.items())
	elif isinstance(d, torch.Tensor) or hasattr(d, 'to'):
		#return d.to(device, non_blocking=True)
		return d.to(device, **kwargs)
	else:
		return d

def collate_fn(batch, pad_id, key=None):
	# this function is designed to support any customized type and to be compatible
	# with the default collate function
	ele = batch[0]

	if isinstance(ele, dict):
		return {key: collate_fn([d[key] for d in batch], pad_id, key) for key in ele}
	elif isinstance(ele, (tuple, list)):
		res = [collate_fn(x, pad_id) for x in zip(*batch)]
		return res
	else:
		if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
			#print(key, [b.shape for b in batch])
			if not all(b.shape == batch[0].shape for b in batch[1:]):
				assert all(len(b.shape) == len(batch[0].shape) for b in batch[1:])
				shape = torch.tensor([b.shape for b in batch])
				max_shape = tuple(shape.max(dim=0)[0].tolist())
				batch2 = []
				for b in batch:
					if any(c < m for c, m in zip(b.shape, max_shape)):
						b2 = torch.ones(max_shape, dtype=b.dtype, device=b.device) * pad_id
						if b.dim() == 1:
							b2[:b.shape[0]] = b
						elif b.dim() == 2:
							b2[:b.shape[0], :b.shape[1]] = b
						elif b.dim() == 3:
							b2[:b.shape[0], :b.shape[1], :b.shape[2]] = b
						else:
							raise NotImplementedError
						b = b2
					batch2.append(b)
				batch = batch2
		return default_collate(batch)

def loadData(nlpData, tokenizer, params, otherData):
	tokenizer.transform(nlpData, params['global']['max_tokens'])

	all_data = []
	for j in tqdm(range(len(nlpData))):

		(binPath, vaddr, realFunctionName, functionName, tokens, bId, fId) = nlpData[j]  # [binPath, vaddr, real_name, name, vaddr, bId, fId]

		# remove unknown tokens and pad again
		tokens = [t for t in tokens if t != tokenizer.unk_token_id]
		while len(tokens) < params['global']['max_tokens'] + 2:
			tokens += [0]

		target = torch.tensor(tokens)
		need_predict = torch.tensor([0] + [1] * (len(target)-1) ) # sos target eos

		data = {
			'caption_tokens': target,
			'need_predict': need_predict,
			'caption': {},
			'iteration': 0,
		}

		assert not np.any(np.isnan(data["caption_tokens"].numpy()))
		assert not np.any(np.isnan(data["need_predict"].numpy()))

		data['caption_hash'] =  int(hashlib.sha256(data["caption_tokens"].numpy().data).hexdigest(), 16) % (10 ** 8)

		for (name, key, d) in otherData:
			if name in ["clap"]:
				if (binPath, vaddr) not in d.keys():
					data[name] =  torch.zeros(768, dtype = torch.float)
					continue

			elif(key and (binPath, vaddr, realFunctionName) not in d.keys()):
				if name == "palmtree":
					data["palmtree"] =  torch.zeros(1, params['COMBO']['PalmtreeSeq']['emb_size'], dtype = torch.float)
					continue
				else:
					data["dexter"] =  torch.zeros(512, dtype = torch.float)
					continue

			if key == True:
				if name in ["clap"]:
					data[name] = d[(binPath, vaddr)]
				else:
					data[name] = d[(binPath, vaddr, realFunctionName)]
			else:
				data[name] = d[fId]

			if name == "palmtree":
				data[name] = [emb for addr, emb in data[name]][:params['COMBO']['PalmtreeSeq']['size']]
				if len(data[name]) == 0:
					data["palmtree"] = torch.zeros(1, params['COMBO']['PalmtreeSeq']['emb_size'], dtype = torch.float)
				else:
					data["palmtree"] = torch.stack(data[name])

			elif type(data[name]) is np.ndarray:
				assert not np.any(np.isnan(data[name]))
				data[name] = torch.tensor(data[name], dtype = torch.float)

		for x in data:
			if torch.is_tensor(data[x]):
				data[x] = data[x].to(torch.device("cpu"))
		all_data.append(data)

	data = collate_fn(all_data, tokenizer.pad_token_id)

	return data


def restrictData(data, indices, cuda=True, combo=False):
	# Remove duplicates (important for COMBO contrastive loss)
	if combo:
		indicesSelected = []
		captions = {}
		for i in indices.tolist():
			if data['caption_hash'][i] in captions:
				continue
			indicesSelected += [i]
			captions[data['caption_hash'][i]] = True
		indices = torch.tensor(np.array(indicesSelected))

	d2 = {}
	for n in data:
		if n == 'caption_hash':
			continue

		if isinstance(data[n], torch.Tensor):
			if data[n].is_sparse:
				d2[n] = data[n].index_select(0, indices).to_dense()
			else:
				d2[n] = data[n][indices]
		else:
			d2[n] = data[n]
	if cuda:
		d2 = recursive_to_device(d2, 'cuda')
	return d2

