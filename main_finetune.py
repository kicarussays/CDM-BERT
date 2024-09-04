from src.packages import *
from src.utils import *
from src.model import *
from src.dataset import BinaryOutcomeDataset
from src.trainer import EHRClassifier

torch.set_num_threads(8)


parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default='cpu')
parser.add_argument('--bs', type=int, 
                    help='Batch size', default=16)
parser.add_argument('--lr', type=float, 
                    help='Learning rate', default=5e-5)
parser.add_argument('--max-epoch', type=int, 
                    help='Max epoch', default=100)
parser.add_argument('--seed', type=int, 
                    help='Random seed', default=50)
parser.add_argument('--use-label-type', '-l', type=int, 
                    help='Use label type?', default=1)
parser.add_argument('--drug', type=str, 
                    help='Adverse event')
parser.add_argument('--cond', type=str, 
                    help='Adverse event')
parser.add_argument('--pretrained', '-p', type=int, 
                    help='Use pretrained model', default=0)
parser.add_argument('--week', '-w', type=int, 
                    help='Weeks before adverse event', default=4)
parser.add_argument('--mlm-const', '-m', type=int, 
                    help='MLM token construction mode', default=0)
parser.add_argument('--hosp', type=str, 
                    help='snuh or ajou', default='snuh')
parser.add_argument('--pre-from', type=str, 
                    help='Use which pretrained model?', default='snuh')
parser.add_argument('--pre-ext', type=str, 
                    help='Use which external pretrained model?', default='snuh')

args = parser.parse_args()

def main():
    seedset(args.seed)
    dc = f'{args.drug}_{args.cond}'
    params = f'w{args.week}_l{args.use_label_type}_p{args.pretrained}'
    os.makedirs(f'../results/finetune_{args.hosp}_{args.pre_from}_{args.pre_ext}/logs/{dc}/{params}/', exist_ok=True)
    os.makedirs(f'../results/finetune_{args.hosp}_{args.pre_from}_{args.pre_ext}/saved/{dc}/{params}/', exist_ok=True)
    os.makedirs(f'../usedata/mlm/{dc}/', exist_ok=True)

    args.logpth = f'../results/finetune_{args.hosp}_{args.pre_from}_{args.pre_ext}/logs/{dc}/{params}/finetune_seed{args.seed}.log'
    args.savepth = f'../results/finetune_{args.hosp}_{args.pre_from}_{args.pre_ext}/saved/{dc}/{params}/finetune_ckpt_seed{args.seed}.tar'

    vocab = torch.load(f'../usedata/tokens/vocabulary_total.pt')
    loadmlm = {}
    for file in ('train', 'valid', 'test'):
        fpath = f'../usedata/mlm/{dc}/{file}_week{args.week}.pkl'
        if os.path.exists(fpath):
            if not args.mlm_const:
                loadmlm[file] = pickleload(fpath)
        else:
            _case = pickleload(f'../usedata/tokens/{dc}/{file}_case_week{args.week}.pkl')
            _cont = pickleload(f'../usedata/tokens/{dc}/{file}_control_week{args.week}.pkl')
            token = {}
            for k in _case.keys():
                token[k] = torch.cat([_case[k], _cont[k]])
            label = torch.Tensor([1] * len(_case[k]) + [0] * len(_cont[k]))

            seg = token['segment']
            mins = seg[:, 1] - 1
            mins = mins.expand((2048,) + mins.size()).T
            token['segment'] = torch.clamp(seg - mins, min=0)

            age_ok = []
            for n, i in enumerate(token['age'][:, 1]):
                if i >= 0:
                    age_ok.append(n)
            for k, v in token.items():
                token[k] = v[age_ok]
            label = label[age_ok]

            dataset = BinaryOutcomeDataset(token, 
                                           outcomes=label,
                                           vocabulary=vocab, 
                                           ignore_special_tokens=True)
            picklesave(dataset, fpath)
            loadmlm[file] = dataset            
            
    if not args.mlm_const:
        bertconfig = BertConfig(
                vocab_size=50000,
                type_vocab_size=1024,
                max_position_embeddings=2048, # used for sequence embeddings
                hidden_size=256,
                num_hidden_layers=6,
                linear=True,
                num_attention_heads=8,
                intermediate_size=512,
                problem_type='single_label_classification'
            )
        
        model = EHRBertForSequenceClassification(bertconfig)
        if args.pretrained:
            device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
            load_pretrained = torch.load(f'../results/pretrain_{args.pre_from}_{args.pre_ext}/saved/m0.3_l{args.use_label_type}/pretrain_ckpt_seed100.tar',
                                        map_location=device)
            model.load_state_dict(load_pretrained['model'], strict=False)

        Trainer = EHRClassifier(
            model=model,
            train_dataset=loadmlm['train'],
            valid_dataset=loadmlm['valid'],
            test_dataset=loadmlm['test'],
            args=args
        )
        Trainer.train()


if __name__ == '__main__':
    main()