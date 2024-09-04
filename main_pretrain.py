from src.packages import *
from src.utils import *
from src.model import *
from src.dataset import MLM_Dataset
from src.trainer import EHRTrainer

torch.set_num_threads(32)


parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default=0)
parser.add_argument('--bs', type=int, 
                    help='Batch size', default=16)
parser.add_argument('--lr', type=float, 
                    help='Learning rate', default=5e-5)
parser.add_argument('--max-epoch', type=int, 
                    help='Max epoch', default=100)
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=100)
parser.add_argument('--masked-ratio', '-m', type=float, 
                    help='Masking rate', default=0.3)
parser.add_argument('--use-label-type', '-l', type=int, 
                    help='Use label type?', default=1)
parser.add_argument('--hosp', type=str, 
                    help='snuh or ajou', default='ajou')
parser.add_argument('--external', '-e', type=str, 
                    help='Use which pretrained model?', default='ajou')
parser.add_argument('--multi-gpu', type=int, 
                    help='Use multi gpu?', default=0)

# argument for multi-gpu
parser.add_argument('--num_workers', type=int, default=16, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

args = parser.parse_args()

if args.multi_gpu:
    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size


def main():
    seedset(args.seed)
    params = f'm{args.masked_ratio}_l{args.use_label_type}'
    os.makedirs(f'../results/pretrain_{args.hosp}_{args.external}/logs/{params}/', exist_ok=True)
    os.makedirs(f'../results/pretrain_{args.hosp}_{args.external}/saved/{params}/', exist_ok=True)
    os.makedirs(f'../usedata/mlm/pretrain_{args.hosp}/', exist_ok=True)
    
    args.logpth = f'../results/pretrain_{args.hosp}_{args.external}/logs/{params}/pretrain_seed{args.seed}.log'
    args.savepth = f'../results/pretrain_{args.hosp}_{args.external}/saved/{params}/pretrain_ckpt_seed{args.seed}.tar'

    vocab = torch.load(f'../usedata/tokens/vocabulary_total.pt')
    loadmlm = {}
    for file in ('train', 'valid'):
        fpath = f'../usedata/mlm/pretrain_{args.hosp}/{file}_{args.masked_ratio}.pkl'
        if os.path.exists(fpath):
            loadmlm[file] = pickleload(fpath)
        else:
            token = pickleload(f'../usedata/tokens/pretrain_{args.hosp}/{file}_token.pkl')
            seg = token['segment']
            mins = seg[:, 1] - 1
            mins = mins.expand((2048,) + mins.size()).T
            token['segment'] = torch.clamp(seg - mins, min=0)
            
            # index with negative age
            age_error = token['age'][:, 1] >= 0
            for k, v in token.items():
                token[k] = v[age_error]

            # for k, v in token.items():
            #     token[k] = v[:1000]

            dataset = MLM_Dataset(token, 
                                  vocabulary=vocab, 
                                  masked_ratio=args.masked_ratio,
                                  ignore_special_tokens=True)
            
            picklesave(dataset, fpath)
            loadmlm[file] = dataset
            
            
    bertconfig = BertConfig(
            vocab_size=50000,
            type_vocab_size=1024,
            max_position_embeddings=2048, # used for sequence embeddings
            hidden_size=256,
            num_hidden_layers=6,
            linear=True,
            num_attention_heads=8,
            intermediate_size=512
        )
    
    model = EHRBertForMaskedLM(bertconfig)
    Trainer = EHRTrainer(
        model=model,
        train_dataset=loadmlm['train'],
        valid_dataset=loadmlm['valid'],
        args=args
    )

    if args.multi_gpu:
        mp.spawn(Trainer.train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        Trainer.train()


if __name__ == '__main__':
    main()