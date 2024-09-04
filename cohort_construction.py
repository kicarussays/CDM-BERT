from src.packages import *
from src.utils import *
from src.dataset import *
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--drug', '-d', type=str, 
                    help='Drug group to construct', default='NSAIDs')
parser.add_argument('--week', '-w', type=int, 
                    help='Weeks before adverse event', default=4)

args = parser.parse_args()


vocab = torch.load('../usedata/tokens/vocabulary_total.pt')
vocab = [i.split('_')[0] for i in list(vocab.keys())[7:]]

person = pd.read_csv('../ehr_file/person.csv')
person = person.set_index("person_id")
concept = pd.read_csv('../ehr_file/CONCEPT.csv', delimiter='\t')
concept = concept[concept['concept_id'].astype(str).isin(vocab)]

concept = concept[concept['concept_name'].notnull()]
concept['concept_name'] = concept['concept_name'].str.lower()
concept_d = concept[(concept['domain_id'] == 'Drug') & (concept['vocabulary_id'].isin(['RxNorm', 'RxNorm Extension']))]
concept_m = concept[(concept['domain_id'] == 'Measurement') & (concept['vocabulary_id'] == 'LOINC')]
concept_c = concept[(concept['domain_id'] == 'Condition') & (concept['vocabulary_id'] == 'SNOMED')]
concept_p = concept[(concept['domain_id'] == 'Procedure') & (concept['vocabulary_id'] == 'SNOMED')]
ca = pd.read_csv('../ehr_file/CONCEPT_ANCESTOR.csv', delimiter='\t')
icdmap = pd.read_csv('../ehr_file/icd_mapped.csv')
icdmap['target_concept_id'] = icdmap['target_concept_id'].astype(int)


all_dict = {}
for i in tqdm(range(20)):
    _dict = pickleload(f'../usedata/code_embedding/all_code_{i}.pkl')
    
    for k, v in _dict.items():
        all_dict[k] = v.dropna()

from sklearn.model_selection import train_test_split
allpid = list(all_dict.keys())
train_id, test_id = train_test_split(allpid, test_size=0.2, random_state=50) # Train:Val:Test = 7:1:2
train_id, val_id = train_test_split(train_id, test_size=0.125, random_state=50)


KEYWORDS = {
    # Diags
    'ICH': 'intrac.*hemo|cereb.*hemo|brain.*hemo',

    # Drugs
    'Anticoagulant': 'rivaroxaban|apixaban|edoxaban|dabigatran|warfarin|aspirin|heparin|enoxaparin|dalteparin',
    'NOACs': 'rivaroxaban|apixaban|edoxaban|dabigatran',
    'NSAIDs': 'aspirin|diclofenac|aceclofenac|indomethacin|ibuprofen|naproxen|celecoxib|flurbiprofen|fenoprofen|ketoprofen|loxoprofen|oxaprozine|piroxicam|meloxicam',
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3726790/pdf/kjfm-34-228.pdf
    'Antidepressant': 'fluoxetine|sertraline|paroxetine|citalopram|amitriptyline|imipramine|nortriptyline|duloxetine|venlafaxine',
    # Chemotherapy (http://www.health.kr/Menu.PharmReview/_uploadfiles/%ED%95%AD%EC%95%94%EC%A0%9C.pdf)
    'Alkylating_agent': 'cyclophosphamide|bendamustine|melphalan|cisplatin|carboplatin|oxaliplatin|busulfan|dacarbazine|temozolomide',
    'Antimetabolites': 'fluorouracil|capecitabine|doxifluridine|tegafur|azacitidine|decitabine|enocitabine|methotrexate|pemetrexed|pralatrexate|cladribine|fludarabine|mercaptopurine',
    'Topoisomerase_inhibitors': 'doxorubicin|daunorubicin|epirubicin|idarubicin|mitoxantrone|etoposide|irinotecan|topotecan',
    'Microtubule_targeting_agents': 'cabazitaxel|paclitaxel|docetaxel|vinblastine|vincristine|vinorelbine',
    'Glucocorticoid': 'betamethasone|dexamethasone|methylprednisolone|prednisolone|triamcinolone|hydrocortisone|deflazacort',
    'Sulfhydryl_containing_agents': 'alacepril|captopril|zofenopril',
    'Dicarboxylate_containing_agents': 'enalapril|ramipril|perindopril|lisinopril|benazepril|imidapril|cilazapril',
    'Phosphonate_containing_agents': 'fosinopril'
}

KEYWORDS['Chemotherapy'] = f"{KEYWORDS['Alkylating_agent']}|{KEYWORDS['Antimetabolites']}|{KEYWORDS['Topoisomerase_inhibitors']}|{KEYWORDS['Microtubule_targeting_agents']}"
KEYWORDS['ACE'] = f"{KEYWORDS['Sulfhydryl_containing_agents']}|{KEYWORDS['Dicarboxylate_containing_agents']}|{KEYWORDS['Phosphonate_containing_agents']}"

# ICH
CIDS = {
    'ICH': pd.unique(concept_c[concept_c['concept_name'].str.contains(KEYWORDS['ICH'])]['concept_id']),
    'PU': np.array([
        198798, 442270, 4027663, 4028242, 4057060, 4057074, 4059178, 4134146, 4198381, 4215084, 
        4265600, 4271696, 4318534, 4319441, 4340787, 36683388, 37110314, 42538071, 45757062,
        4027729, 4046500, 4174044, 4211001, 4231580, 4232181,
        4057953, 4146517, 4150681, 4173408, 4194543, 4265479, 4338225
    ]),
    'NF': np.array([4250734, 3191245]),
    'OP': pd.unique(concept_c[concept_c['concept_name'].str.contains('osteoporosis')]['concept_id']),
}

CIDS = {
    k: pd.unique(ca[ca['ancestor_concept_id'].isin(v)]['descendant_concept_id'])
    for k, v in CIDS.items()
}

def drug_split(category):
    return dict({
        # noac + warfarin + aspirin + heparin + lmwh
        category: drug_id(
            pd.unique(concept_d[concept_d['concept_name'].str.contains(KEYWORDS[category])]['concept_id'])),
    },**{
        d: drug_id(
            pd.unique(concept_d[concept_d['concept_name'].str.contains(d)]['concept_id'])
        ) for d in KEYWORDS[category].split('|')
    })

DIDS = {}
for category in ('NSAIDs', 
                 'Anticoagulant', 'NOACs', 
                 'Glucocorticoid',
                 'Chemotherapy', 'Alkylating_agent', 'Antimetabolites', 'Topoisomerase_inhibitors', 'Microtubule_targeting_agents',):
    DIDS = dict(DIDS, **drug_split(category))

def casecontrol(cond, drug, week, pids):
    allk = []
    for pid in tqdm(pids):
        codes = all_dict[pid]
        if codes[codes['code'].isin(DIDS[drug])].shape[0] > 0:
            allk.append(pid)

    cohort = {'case': {}, 'control': {}}
    for pid in tqdm(allk):
        codes = all_dict[pid]

        ccodes = codes[codes['code'].isin(CIDS[cond])]
        if ccodes.shape[0] > 0:
            dcodes = codes[codes['code'].isin(DIDS[drug])]
            dcodes['diff'] = (ccodes['date'].min() - dcodes['date']).dt.days
            day_diff = dcodes[dcodes['diff'] >= 0]['diff'].min()
            if day_diff <= week*7:
                
                until = ccodes['date'].min() - timedelta(days=week*7)
                finalcohort = codes[codes['date'] < until]
                if finalcohort.shape[0] >= 50:
                    cohort['case'][pid] = finalcohort
        
        else:
            # lastidx = codes[codes['code'].isin(DIDS[drug])].index[-1]
            lastidx = codes[codes['code'].isin(DIDS[drug])].sample(1).index[0]
            finalcohort = codes.loc[:lastidx]
            if finalcohort.shape[0] >= 50:
                cohort['control'][pid] = finalcohort
    
    return cohort


from src.tokenizer import EHRTokenizer
tokenizer_config = DotDict({
    'sep_tokens': True, # should we add [SEP] tokens?
    'cls_token': True, # should we add a [CLS] token?   
    'padding': True, # should we pad the sequences?
    'truncation': 2048}) # how long should the longest sequence be
vocab = torch.load(f'../usedata/tokens/vocabulary_total.pt')
tokenizer = EHRTokenizer(vocabulary=vocab, config=tokenizer_config)
tokenizer.freeze_vocabulary()

def cohort_tokenization(allvar):
    train_id, val_id, test_id, cond, drug, week = allvar
    os.makedirs(f'../usedata/tokens/{drug}_{cond}/', exist_ok=True)

    print('Case-control screening...')
    cohort = {}
    cohort['train'] = casecontrol(cond, drug, week, train_id)
    cohort['valid'] = casecontrol(cond, drug, week, val_id)
    cohort['test'] = casecontrol(cond, drug, week, test_id)
    print('Done')
    
    for t, c in cohort.items():
        for k, udict in c.items():
            print(f'Constructing << cohort_{drug}_{cond}_{t}_{k} >>...')
            features = {}
            concept, age, segment, label = [], [], [], []
            ulen = 2040
            for id in tqdm(udict.keys()):
                if udict[id]['age'].iloc[-ulen:].min() >= 18:
                    pg = person.loc[id]['gender_source_value']
                    concept.append(
                        [pg] + list(udict[id]['code'].astype(str).values)[-ulen:])
                    age.append(
                        [udict[id]['age'].iloc[-ulen:].iloc[0]] + list(udict[id]['age'].values)[-ulen:])
                    segment.append(
                        [udict[id]['segment'].iloc[-ulen:].iloc[0]] + list(udict[id]['segment'].values[-ulen:] - min(udict[id]['segment'].values[-ulen:]) + 1))
                    label.append(
                        [0] + list(udict[id]['label'].values)[-ulen:])

            features['concept'] = concept
            features['age'] = age
            features['segment'] = segment
            features['label'] = label
            tokenized = tokenizer(features)
            picklesave(tokenized, f'../usedata/tokens/{drug}_{cond}/{t}_{k}_week{week}.pkl')
            del tokenized, concept, age, segment, label; gc.collect()
            


idx_list = {
    'NSAIDs': [
        ('NSAIDs', 'PU'),
        ('aspirin', 'PU'),
        ('celecoxib', 'PU'),
        ('aceclofenac', 'PU'),
        ('ibuprofen', 'PU'),
    ],

    'Anticoagulant': [
        ('Anticoagulant', 'ICH'),
        ('heparin', 'ICH'),
        ('warfarin', 'ICH'),
        # ('enoxaparin', 'ICH'),
        # ('NOACs', 'ICH'),
    ],

    'Glucocorticoid': [
        ('Glucocorticoid', 'OP'),
        ('dexamethasone', 'OP'),
        ('methylprednisolone', 'OP'),
        ('prednisolone', 'OP'),
        ('hydrocortisone', 'OP'),
    ],

    'Chemotherapy': [
        ('Chemotherapy', 'NF'),
        ('Alkylating_agent', 'NF'),
        ('Antimetabolites', 'NF'),
        ('Topoisomerase_inhibitors', 'NF'),
        ('Microtubule_targeting_agents', 'NF'),
    ],
}

import multiprocessing
all_vars = []
for k, v in idx_list.items():
    for i in v:
        if i[0] == k:
            for w in (2, 4, 8, 12):
                all_vars.append([train_id, val_id, test_id, i[1], i[0], w])
        else:
            all_vars.append([train_id, val_id, test_id, i[1], i[0], 4])

pool = multiprocessing.Pool(16)
pool.map(cohort_tokenization, all_vars)
pool.close()
pool.join

