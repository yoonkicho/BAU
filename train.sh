# Protocol-2
CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501 msmt17 cuhksysu -dt cuhk03 -b 256 --lam 1.5 --k 10 --iters 500
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501 cuhksysu cuhk03 -dt msmt17 -b 256 --lam 1.5 --k 10 --iters 200
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds msmt17 cuhk03 cuhksysu -dt market1501 -b 256 --lam 1.5 --k 10 --iters 500

# Protocol-3
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501dg msmt17dg cuhksysu -dt cuhk03 -b 256 --lam 1.5 --k 10 --iters 1000
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501dg cuhksysu cuhk03dg -dt msmt17 -b 256 --lam 1.5 --k 10 --iters 400
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds msmt17dg cuhk03dg cuhksysu -dt market1501 -b 256 --lam 1.5 --k 10 --iters 1000

# Protocol-1
# CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py -a resnet50 -ds market1501dg cuhk02dg cuhk03dg cuhksysu -dt grid -b 256 --lam 1.5 --k 10 --iters 500
