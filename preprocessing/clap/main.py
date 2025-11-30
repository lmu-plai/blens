import os
import pickle
import argparse

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def restrictData(data, indices, dev):
    d2 = {}
    for n in data:
        d2[n] = data[n][indices].to(dev)
    return d2

def main(pathNlpData, pathIdaB, pathScript, pathOut, dev, batch_size):
	# Select binaries
	with open(pathNlpData, 'rb') as f:
		nlpData = pickle.load(f)
	
	exportedEmbeddings = {}
	binaries = set()
	for subset in nlpData:
		for (binPath, vaddr, realFunctionName, functionName, vaddr, bId, fId) in subset:
			binaries.add(binPath)
			exportedEmbeddings[(binPath, vaddr)] = True

	# CLAP models
	asm_tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
	asm_encoder = AutoModel.from_pretrained("hustcw/clap-asm", trust_remote_code=True).to(dev)

	# Run IDA and compute CLAP embeddings
	embeddings = {}

	for binPath in binaries:
		
		# Run CLAP's IDA Analysis
		os.system(pathIdaB+" -A -c -S\""+pathScript+"\" "+binPath+" >/dev/null 2>&1")

		pathResult = binPath+'-clap.pkl'
		if os.path.isfile(pathResult) == False:
			print("Error at", _binPath)
			continue

		try:
			# Compute CLAP embeddings
			with open(pathResult, "rb") as f:
				disasm = pickle.load(f)
			
			# Remove functions outside of nlpData
			disasmCleaned = []
			for (vaddr, rebased_assembly) in disasm:
				if (binPath, vaddr) in exportedEmbeddings:					
					disasmCleaned+= [[vaddr, rebased_assembly]]
			disasm = disasmCleaned
			
			size = len(disasm)
			clapEncoding = []
			for f in range(size):
				clapEncoding += [disasm[f][1]]
			
			asm_input = asm_tokenizer(clapEncoding, padding=True, return_tensors="pt")
			
			idx = 0	
			for i in tqdm(range(0, size, batch_size)):
				batchIds = torch.from_numpy( np.arange(i, min(  i + batch_size, size)))
				bs = batchIds.shape[0]
				batch = restrictData(asm_input, batchIds, dev)

				with torch.no_grad():
					asm_embedding = asm_encoder(**batch).detach().cpu().numpy()
					for j in range(bs):
						embeddings[(binPath, disasm[idx][0])] = asm_embedding[j]
						idx += 1

		except Exception as e:
			print(e)

	with open(pathOut, "wb") as f:
		pickle.dump(embeddings, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute CLAP embeddings from a nlpData file ")
    parser.add_argument('--nlpData', type=str, required=True, 
                        help='Absolute path to the NLP data directory')
    parser.add_argument('--idaB', type=str, required=True, 
                        help='Absolute path to the IDA Pro binary, e.g. /opt/idapro-7.6/idat64')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path')

    parser.add_argument('--cpu', action='store_true', help='Disable CUDA')
    parser.add_argument('-batch-size', '--batch-size', default=32, type=int)
    
    args = parser.parse_args()
    
    idaScript = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'idaScript.py')
    dev = torch.device('cpu') if args.cpu else  torch.device('cuda')
    
    print('CLAP Embeddings')
    main(args.nlpData, args.idaB, idaScript, args.output, dev, args.batch_size)
