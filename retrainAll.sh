# Config
DEVICE=0
DATAFOLDER="PATH-TO-DATA-FOLDER"

# Main Results

## Cross-binary settings

# BLens (96 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=cb-bl --cross-binary -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=cb-bl -name=cb-bl --cross-binary > cb-bl.txt
cd ../

# BL-S (96 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=cb-bl-s --cross-binary --symlm-subdataset -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=cb-bl-s -name=cb-bl-s --cross-binary --symlm-subdataset > cb-bl-s.txt
cd ../

# BL-A (96 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=cb-bl-a --cross-binary --asmdepictor-subdataset -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=cb-bl-a -name=cb-bl-a --cross-binary --asmdepictor-subdataset > cb-bl-a.txt
cd ../

## Cross-project settings

# BLens (96 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=cp-bl --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=cp-bl -name=cp-bl --cross-project > cp-bl.txt
cd ../

# BL-S (96 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=cp-bl-s --cross-project --symlm-subdataset -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=cp-bl-s -name=cp-bl-s --cross-project --symlm-subdataset > cp-bl-s.txt
cd ../

# BL-A (96 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=cp-bl-a --cross-project --asmdepictor-subdataset -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=cp-bl-a -name=cp-bl-a --cross-project --asmdepictor-subdataset > cp-bl-a.txt
cd ../

# Ablations

# BLens ablation base model (BLens, C+P+D, LORD) + LORD-T0 + COMBO as a decoder /  (66 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-base -config=ablation-base.json --cross-project -pretrain -train -inferBest -reinferAllNoThreshold -inferCOMBO
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-base -name=ablation-base --cross-project > ablation-base.txt
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-base -name=ablation-lord-t0 --cross-project --evaluateNoThreshold > ablation-lord-t0.txt
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-base -name=ablation-combo --cross-project --evaluateComboDecoder > ablation-combo.txt
cd ../

# BL-NP  (54 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-np  -config=ablation-np.json --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-np -name=ablation-np --cross-project > ablation-np.txt
cd ../

# C (60 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-c  -config=ablation-c.json --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-c -name=ablation-c --cross-project > ablation-c.txt
cd ../

# P (60 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-p  -config=ablation-p.json --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-p -name=ablation-p --cross-project > ablation-p.txt
cd ../

# D (60 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-d  -config=ablation-d.json --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-d -name=ablation-d --cross-project > ablation-d.txt
cd ../

# P+D  (60 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-p+d  -config=ablation-p+d.json --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-p+d -name=ablation-p+d --cross-project > ablation-p+d.txt
cd ../

# C+P  (60 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-c+p  -config=ablation-c+p.json --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-c+p -name=ablation-c+p --cross-project > ablation-c+p.txt
cd ../

# C+D  (60 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-c+d  -config=ablation-c+d.json --cross-project -pretrain -train -inferBest
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-c+d -name=ablation-c+d --cross-project > ablation-c+d.txt
cd ../

# SIMPLE + SIMPLE-T0  (64 hours)
CUDA_VISIBLE_DEVICES=$DEVICE python3 RunExp.py -data-dir="$DATAFOLDER" -d=ablation-simple -config=ablation-simple.json --cross-project -pretrain -train -inferBest -reinferAllNoThreshold
cd evaluation/
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-simple -name=ablation-simple --cross-project > ablation-simple.txt
python3 new_experiments_evaluator.py -data-dir="$DATAFOLDER" -d=ablation-simple -name=ablation-simple-t0 --cross-project --evaluateNoThreshold > ablation-simple-t0.txt
cd ../


