import idc
import idaapi
import idautils
import ida_pro
import ida_auto
import ida_nalt

from copy import deepcopy
import pickle

loc_pattern = re.compile(r' (loc|locret)_(\w+)')
self_pattern = re.compile(r'\$\+(\w+)')

def rebase(asm_dict):	
	asm_dict = deepcopy(asm_dict)
	
	index = 1
	rebase_assembly = {}

	addrs = list(sorted(list(asm_dict.keys())))

	for addr in addrs:
		inst = asm_dict[addr]
		if inst.startswith('j'):
			loc = loc_pattern.findall(inst)
			for prefix, target_addr in loc:
				try:
					target_instr_idx = addrs.index(int(target_addr, 16)) + 1
					asm_dict[addr] = asm_dict[addr].replace(f' {prefix}_{target_addr}', f' INSTR{target_instr_idx}')
				except Exception:
					continue

			self_m = self_pattern.findall(inst)
			for offset in self_m:
				try:
					target_instr_addr = addr + int(offset, 16)
					target_instr_idx = addrs.index(target_instr_addr)
					asm_dict[addr] = asm_dict[addr].replace(f'$+{offset}', f'INSTR{target_instr_idx}')
				except:
					continue

		rebase_assembly[str(index)] = asm_dict[addr]
		index += 1

	return rebase_assembly

def main():

	textStartEA = 0
	textEndEA = 0
	for seg in idautils.Segments():
		if (idc.get_segm_name(seg)==".text"):
			textStartEA = idc.get_segm_start(seg)
			textEndEA = idc.get_segm_end(seg)
			break
	
	functions = []
	for function_ea in idautils.Functions(textStartEA, textEndEA):

		flags = idc.get_func_attr(function_ea, idc.FUNCATTR_FLAGS)
		if flags & idc.FUNC_LIB:
			continue
	
		instGenerator = idautils.FuncItems(function_ea)
		raw_assembly = {}
		for inst in instGenerator:
			raw_assembly[inst] = idc.GetDisasm(inst)
		rebased_assembly = rebase(raw_assembly)

		functions+= [[function_ea, rebased_assembly]]

	with open(f'{ida_nalt.get_input_file_path()}-clap.pkl', 'wb') as f:
		pickle.dump(functions, f)

if __name__ == '__main__':
	ida_auto.auto_wait()
	main()
	ida_pro.qexit(0)
