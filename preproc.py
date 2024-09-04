from src.packages import *
from src.utils import *
from src.vars import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=0)
args = parser.parse_args()

# ray.init(num_cpus=48)

# # Drug code setting
# if not os.path.exists('../ehr_file/drug_code.csv'):
#     drug = mpd.read_csv('../ehr_file/drug.csv')
#     drug['drug_concept_id'] = drug['drug_concept_id'].astype(str)
#     concept = mpd.read_csv('../ehr_file/CONCEPT.csv', delimiter='\t')
#     concept['concept_id'] = concept['concept_id'].astype(str)

#     drug = drug[drug['drug_concept_id'].isin(
#         mpd.unique(concept[(concept['domain_id'] == 'Drug')
#                           & (concept['vocabulary_id'].str.contains('RxNorm'))]['concept_id'])
#     )]
#     did = mpd.unique(drug['drug_concept_id'])

#     drug['drug_exposure_start_date'] = mpd.to_datetime(drug['drug_exposure_start_date']).dt.date
#     drug['diff'] = (mpd.to_datetime(drug['drug_exposure_end_date']) - mpd.to_datetime(drug['drug_exposure_start_date'])).dt.days + 1
#     drug['dose'] = drug['diff'].apply(dose_dummy)
#     drug['drug_codes'] = drug.apply(lambda x: f"{x['drug_concept_id']}_{x['dose']}", axis=1)
#     drug_ids = mpd.unique(drug['drug_concept_id'])
#     drug_codes = np.array([f'{i}_short' for i in drug_ids] + [f'{i}_long' for i in drug_ids])
#     drug.to_csv('../../ehr_file/drug_code.csv', index=None)


# visit = mpd.read_csv('../../ehr_file/visit.csv')
# meas = mpd.read_csv('../../ehr_file/measure.csv')


# # Split due to memory issue
# pids = np.array_split(mpd.unique(visit['person_id']), 10)
# for n, pid in enumerate(pids):
#     meas[meas['person_id'].isin(pid)].to_csv(f'../../ehr_file/measure_{n}.csv', index=None)

# for i in range(1, 10):
#     print(i)
#     meas = mpd.read_csv(f'../../ehr_file/measure_{i}.csv')
#     meas['measurement_date'] = meas['measurement_datetime'].apply(lambda x: x[:10])
#     meas = meas[meas['measurement_date'].apply(lambda x: int(x[:4]) >= 1990)]
#     meas = meas[meas.columns.difference(['measurement_datetime', 'Unnamed: 0'], sort=False)]
#     tmp = meas.groupby(['person_id', 'measurement_concept_id', 'measurement_date']).cumcount()+1
#     if len(tmp.shape) != 1:
#         meas['nrow'] = tmp.iloc[:, 0]
#     else:
#         meas['nrow'] = tmp

#     meas = meas[meas['nrow'] == 1][meas.columns.difference(['nrow'], sort=False)]
#     meas.to_csv(f'../../ehr_file/measure_{i}.csv', index=None)
#     del meas
#     gc.collect()


# # Measurement code setting
# fpath = '../../ehr_file'
# files = ['cond', 'drug', 'observation', 'person', 'procedure', 'visit', 'stcm']
# allf = {f: mpd.read_csv(os.path.join(fpath, f'{f}.csv')) for f in files}

# meas = [mpd.read_csv(f'../../ehr_file/measure_{i}.csv') for i in range(10)]
# meas = mpd.concat(meas)
# meas = meas[meas['measurement_concept_id'].isin(
#     pd.unique(allf['stcm'][(allf['stcm']['vocabulary'] == 'LOINC') & (allf['stcm']['domain'] == 'Measurement')]['omop_concept_id']))]
# meas = meas[meas['value_as_number'].notnull()]
# mcid = mpd.unique(meas['measurement_concept_id'])
# mcid_count = meas['measurement_concept_id'].value_counts()
# use_mcid = mcid_count[mcid_count>=10000].index
# meas = meas[meas['measurement_concept_id'].isin(list(use_mcid))]
# meas.reset_index(drop=True, inplace=True)

# os.makedirs('../usedata/tmp', exist_ok=True)
# mcid_count = meas['measurement_concept_id'].value_counts()
# use_mcid = mcid_count[mcid_count>=10000].index


# # 10th quantiles of all measurement items
# m_q_path = '../usedata/tmp/m_quantile.pkl'
# if os.path.exists(m_q_path):
#     with open(m_q_path, 'rb') as f:
#         m_quantile = pickle.load(f)
# else:
#     m_quantile = {}
#     for m in tqdm(use_mcid):
#         _mead = meas[meas['measurement_concept_id'] == m]['value_as_number']
#         m_quantile[m] = [_mead.quantile(i/10) for i in range(1, 10)]
#     with open('../usedata/tmp/m_quantile.pkl', 'wb') as f:
#         pickle.dump(m_quantile, f, pickle.HIGHEST_PROTOCOL)

# meas['mcode'] = meas.apply(measure_value_code, axis=1)
# meas.to_csv('../../ehr_file/measure_code.csv', index=None)


# # Final codebook 
# fpath = '../../ehr_file'
# files = ['cond', 'drug_code', 'observation', 'person', 'procedure', 'visit', 'stcm', 'measure_code']
# allf = {f: mpd.read_csv(os.path.join(fpath, f'{f}.csv')) for f in tqdm(files)}

# visit = allf['visit']
# cond = allf['cond']
# drug = allf['drug_code']
# meas = allf['measure_code']
# proc = allf['procedure']
# person = allf['person']

# cond['condition_start_date'] = mpd.to_datetime(cond['condition_start_date']).dt.date
# cond['condition_concept_id'] = cond['condition_concept_id'].astype(str)
# cond_dict = mpd.unique(cond['condition_concept_id'])

# drug['drug_exposure_start_date'] = mpd.to_datetime(drug['drug_exposure_start_date']).dt.date
# drug_ids = mpd.unique(drug['drug_concept_id'])
# drug_codes = np.array([f'{i}_short' for i in drug_ids] + [f'{i}_long' for i in drug_ids])

# proc['procedure_date'] = mpd.to_datetime(proc['procedure_datetime']).dt.date
# proc['procedure_concept_id'] = proc['procedure_concept_id'].astype(str)
# proc_dict = mpd.unique(proc['procedure_concept_id'])

# meas['measurement_date'] = mpd.to_datetime(meas['measurement_date']).dt.date
# meas_ids = mpd.unique(meas['measurement_concept_id'])
# meas_codes = np.concatenate([[f'{i}_{j}' for j in range(10)] for i in meas_ids])
# all_dict = np.concatenate([cond_dict, drug_codes, proc_dict, meas_codes]).astype(str)

# with open('../usedata/alldict.pkl', 'wb') as f:
#     pickle.dump(all_dict, f, pickle.HIGHEST_PROTOCOL)


# # Data split for time reduction
# os.makedirs('../../ehr_file/tmp/', exist_ok=True)
# allpid = mpd.unique(visit['person_id'])
# pidchunk = np.array_split(allpid, 20)
# for n, c in enumerate(pidchunk):
#     cond[cond['person_id'].isin(c)].to_csv(f'../../ehr_file/tmp/cond_{n}.csv', index=None)
#     drug[drug['person_id'].isin(c)].to_csv(f'../../ehr_file/tmp/drug_{n}.csv', index=None)
#     meas[meas['person_id'].isin(c)].to_csv(f'../../ehr_file/tmp/meas_{n}.csv', index=None)
#     proc[proc['person_id'].isin(c)].to_csv(f'../../ehr_file/tmp/proc_{n}.csv', index=None)


# # Code embedding
# os.makedirs('../usedata/code_embedding/', exist_ok=True)
# person = pd.read_csv('../../ehr_file/person.csv')
# for i in range(args.index, args.index+1):
#     all_codes = {}
#     cond = pd.read_csv(f'../../ehr_file/tmp/cond_{i}.csv')
#     drug = pd.read_csv(f'../../ehr_file/tmp/drug_{i}.csv')
#     meas = pd.read_csv(f'../../ehr_file/tmp/meas_{i}.csv')
#     proc = pd.read_csv(f'../../ehr_file/tmp/proc_{i}.csv')

#     pid = np.concatenate([
#         pd.unique(cond['person_id']),
#         pd.unique(drug['person_id']),
#         pd.unique(meas['person_id']),
#         pd.unique(proc['person_id'])
#     ])
#     pid = np.array(list(set(pid)))

#     for p in tqdm(pid):
#         ucond = cond[cond['person_id'] == p][['condition_concept_id', 'condition_start_date']].values
#         umeas = meas[meas['person_id'] == p][['mcode', 'measurement_date']].values
#         udrug = drug[drug['person_id'] == p][['drug_codes', 'drug_exposure_start_date']].values
#         uproc = proc[proc['person_id'] == p][['procedure_concept_id', 'procedure_date']].values
#         ulabel = np.array([1] * len(ucond) + [2] * len(umeas) + [3] * len(udrug) + [4] * len(uproc)) 
#         uall = pd.DataFrame(np.concatenate([
#             ucond, umeas, udrug, uproc
#         ]), columns=['code', 'date'])
#         uall['date'] = mpd.to_datetime(uall['date'])
#         uall['label'] = ulabel
#         uall = uall.sort_values(['date'])

#         pinfo = person[person['person_id'] == p].iloc[0]
#         birth = pd.to_datetime(f"{pinfo['year_of_birth']}-{pinfo['month_of_birth']}-15")
#         uall['age'] = ((uall['date'] - birth).dt.days / 365.25).astype(int)
#         uall['segment'] = uall['date'].rank(method='dense').astype(int)
#         uall.reset_index(drop=True, inplace=True)
#         all_codes[p] = uall

#     with open(f'../usedata/code_embedding/all_code_{i}.pkl', 'wb') as f:
#         pickle.dump(all_codes, f, pickle.HIGHEST_PROTOCOL)


# Tokenize
os.makedirs('../usedata/tokens/', exist_ok=True)
all_dict = {}
for i in tqdm(range(20)):
    _dict = pickleload(f'../usedata/code_embedding/all_code_{i}.pkl')
    
    for k, v in _dict.items():
        all_dict[k] = v.dropna()

# allcode = pd.DataFrame(['[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]', 'F', 'M'], columns=['code'])
# for k, v in tqdm(all_dict.items()):
#     allcode = pd.concat([allcode, v[['code']]]).drop_duplicates()
# allcode.reset_index(drop=True, inplace=True)
# vocab = {str(i): n for n, i in enumerate(allcode['code'])}
# torch.save(vocab, '../usedata/tokens/vocabulary_total.pt')

from sklearn.model_selection import train_test_split
allpid = list(all_dict.keys())
train_id, test_id = train_test_split(allpid, test_size=0.2, random_state=50) # Train:Val:Test = 7:1:2
train_id, val_id = train_test_split(train_id, test_size=0.125, random_state=50)

picklesave({k: all_dict[k] for k in test_id}, 
           '../usedata/code_embedding/all_code_test.pkl')

person = pd.read_csv('../ehr_file/person.csv')
person = person.set_index('person_id')

def get_feature(ids):
    features = {}
    concept, age, segment, label = [], [], [], []
    for id in tqdm(ids):
        pg = person.loc[id]['gender_source_value']
        usel = 2040
        if all_dict[id].shape[0] >= usel:
            _n = all_dict[id].shape[0] // usel
            for _c in range(_n):
                if all_dict[id]['age'].iloc[_c*usel:(_c+1)*usel].min() >= 18:
                    # Add gender
                    concept.append(
                        [pg] + list(all_dict[id]['code'].iloc[_c*usel:(_c+1)*usel].astype(str).values))
                    age.append(
                        [all_dict[id]['age'].iloc[_c*usel]] + list(all_dict[id]['age'].iloc[_c*usel:(_c+1)*usel].values))
                    segment.append(
                        [all_dict[id]['segment'].iloc[_c*usel]] + list(all_dict[id]['segment'].iloc[_c*usel:(_c+1)*usel].values))
                    # 0 for gender label
                    label.append(
                        [0] + list(all_dict[id]['label'].iloc[_c*usel:(_c+1)*usel].values))

            if all_dict[id]['age'].iloc[-usel:].min() >= 18:
                concept.append(
                    [pg] + list(all_dict[id]['code'].iloc[-usel:].astype(str).values))
                age.append(
                    [all_dict[id]['age'].iloc[-usel]] + list(all_dict[id]['age'].iloc[-usel:].values))
                segment.append(
                    [all_dict[id]['segment'].iloc[-usel]] + list(all_dict[id]['segment'].iloc[-usel:].values))
                label.append(
                    [0] + list(all_dict[id]['label'].iloc[-usel:].values))
            
        else:
            if all_dict[id]['age'].min() >= 18:
                concept.append(
                    [pg] + list(all_dict[id]['code'].astype(str).values))
                age.append(
                    [all_dict[id]['age'].iloc[0]] + list(all_dict[id]['age'].values))
                segment.append(
                    [all_dict[id]['segment'].iloc[0]] + list(all_dict[id]['segment'].values))
                label.append(
                    [0] + list(all_dict[id]['label'].values))

    features['concept'] = concept
    features['age'] = age
    features['segment'] = segment
    features['label'] = label

    return features

        
train_features = get_feature(train_id)
val_features = get_feature(val_id)
ut, uv = train_features, val_features

from src.tokenizer import EHRTokenizer
tokenizer_config = DotDict({
    'sep_tokens': True, # should we add [SEP] tokens?
    'cls_token': True, # should we add a [CLS] token?
    'padding': True, # should we pad the sequences?
    'truncation': 2048}) # how long should the longest sequence be

# tokenizer = EHRTokenizer(config=tokenizer_config)
# train_tokenized = tokenizer(ut) 
# tokenizer.freeze_vocabulary()
# tokenizer.save_vocab(f"../usedata/tokens/vocabulary_snuh.pt")
# # Now we tokenize the validation set:
# valid_tokenized = tokenizer(uv)

vocab = torch.load('../usedata/tokens/vocabulary_total.pt')
tokenizer = EHRTokenizer(vocabulary=vocab, config=tokenizer_config)
tokenizer.freeze_vocabulary()
train_tokenized = tokenizer(ut) 
valid_tokenized = tokenizer(uv)

os.makedirs('../usedata/tokens/pretrain/', exist_ok=True)
picklesave(train_tokenized, f'../usedata/tokens/pretrain/train_token.pkl')
picklesave(valid_tokenized, f'../usedata/tokens/pretrain/valid_token.pkl')




