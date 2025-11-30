#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################
# File Name: generate_labels_bb.py
# Author: Yunru Wang
# E-mail: yunruw@outlook.com
# Created Time: 2024-02-16 13:57
# Last Modified: 2024-02-21 17:17
###########################

# Second author: Tristan Benoit on 2025-09-20

import angr
import re
import pickle
import os
import argparse
from tqdm import tqdm

from PalmTree import PalmTreeBasicBlockEncoder

pattern = re.compile(r'.*(\[rip\s*\+\s*(\S+)h\])')
pattern2 = re.compile(r'.*([0-9a-fA-F]{6}h)')


def handle_op(project, op_str, cur_addr, str_addrs):	
	global pattern, pattern2

	op_str = op_str.replace(',', '')
	op_str = op_str.replace('*', ' * ')
	op_str = op_str.replace('ptr', '')
	# [] space and solve mem addr
	if('[' in op_str):
		match = pattern.match(op_str)
		if(match):
			offset = int(match.group(2), 16)
			try:
				cur_state = project.factory.blank_state(addr=cur_addr)
				cur_rip = int(cur_state.regs.rip._model_concrete.value)
			except Exception as e:
				pass
			else:
				mem_addr = cur_rip + offset		
				if(mem_addr in str_addrs):
					op_str = op_str.replace(match.group(1), 'string')

		op_str = op_str.replace('[', '[ ')
		op_str = op_str.replace(']', ' ]')

	# xxxxxxh
	match = pattern2.match(op_str)
	if(match):
		# match of symbol and address
		content = match.group(1)
		symbol = project.loader.find_symbol(int(content[:-1], 16))
		if(symbol):
			op_str = op_str.replace(content, 'symbol')
		else:
			if('0ff' in op_str):
				pass
			else:
				op_str = op_str.replace(content, 'address')

	return op_str



def get_bb_seq(project, func):
	str_addrs = set([int(addr) for addr, string in func.string_references()])
	
	seq = []
	'''
	format should be like:
	["mov rbp rdi", 
		"mov ebx 0x1", 
		"mov rdx rbx", 
		"call memcpy", 
		"mov [ rcx + rbx ] 0x0", 
		"mov rcx rax", 
		"mov [ rax ] 0x2e"]
	'''
	for block in func.blocks:
		bseq = []
		b = block.disassembly
		for ins in b.insns:
			temp_ins = []
			temp_ins.append(ins.mnemonic)
			temp_ins.append(handle_op(project, ins.op_str, ins.address, str_addrs))
			bseq.append(' '.join(temp_ins))
		seq.append((block.addr, bseq))
		if len(seq) >= 50:
			break
	return seq

def basicBlockEmbedding(nlpData, palmtree, cpu):

	binaries = {}
	exportedEmbeddings = {}
	for subset in nlpData:
		for (binPath, vaddr, _, _,  _, _, _) in subset:
			binaries[binPath] = True
			exportedEmbeddings[binPath, vaddr] = True
		
	palmtreeBB = PalmTreeBasicBlockEncoder(os.path.join(palmtree,"pretrained_palmtree"), os.path.join(palmtree,"vocab"), cpu)

	embeddings = {}

	for binPath in binaries:
		
		
		try:			
			project = angr.Project(binPath, load_options={'auto_load_libs': False})
			cfg = project.analyses.CFGFast()
			base_addr = project.loader.main_object.mapped_base
			
			print(binPath, len(cfg.kb.functions))

			for vaddr in tqdm(cfg.kb.functions):
				key = (binPath, vaddr - base_addr)
				if not(key in exportedEmbeddings):
					continue

				try:
					seq = get_bb_seq(project, cfg.kb.functions[vaddr])
				except Exception as e:
					print(e)
				else:
					embeddings[key] = []
					for (vaddrBB, bbSeq) in seq:
						bb_embedding = palmtreeBB.encode(bbSeq)
						embeddings[key] += [(vaddrBB-base_addr, bb_embedding)]

		except Exception as e:
			print(e)
	
	return embeddings


if __name__=="__main__":
	parser = argparse.ArgumentParser(
		description=(
			"PalmTree BasicBlock Embedding Generator for BLens (Angr-based analysis).\n"
			"Input: path to DEXTER nlpData table (python pickle file).\n"
			"Output: Python pickle file with {(binPath, vAddr, func_name): [(vaddrBasicBlock, Embedding)]} dictionary."
		)
	)
	
	parser.add_argument("nlpData", type=str, help="Path to DEXTER 'nlpData' file")
	parser.add_argument("palmtree", type=str, help="Path to PalmTree models")
	parser.add_argument("output", type=str, help="Path to store dictionary of PalmTree Basic Blocks embeddings (pickled)")
	parser.add_argument('--cpu', action='store_true', help='Disable CUDA')
    
	args = parser.parse_args()
	
	with open(args.nlpData, 'rb') as f:
		nlpData = pickle.load(f)
	
	print('PalmTree Basic Bloc Embeddings')
	embeddings = basicBlockEmbedding(nlpData, args.palmtree, args.cpu)
	with open(args.palmtree, 'wb') as f:
		pickle.dump(embeddings, f)
