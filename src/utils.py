from src.packages import *

def dose_dummy(v):
    if v <= 28:
        return 'short'
    else:
        return 'long'


def measure_value_code(x):
    _quantile = m_quantile[x['measurement_concept_id']]
    for n, q in enumerate(_quantile):
        if x['value_as_number'] <= q:
            return f"{x['measurement_concept_id']}_{n}"
    
    if n == len(_quantile):
        return f"{x['measurement_concept_id']}_9"


class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]


def seedset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def picklesave(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)


def pickleload(path):
    with open(path, 'rb') as f:
        tmp = pickle.load(f)
    
    return tmp


def drug_id(ids):
    return [f'{int(id)}_short' for id in ids] + [f'{int(id)}_long' for id in ids]


def casecontrol(cond, drug, pids):
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
            if day_diff <= 30:
                finalcohort = codes[codes['date'] < dcodes[dcodes['diff'] >= 0].iloc[-1]['date']]
                if finalcohort.shape[0] >= 50:
                    cohort['case'][pid] = finalcohort
        
        else:
            cohort['control'][pid] = codes
    
    return cohort


def cohort_tokenization(train_id, val_id, test_id, cond, drug):
    print('Case-control screening...')
    cohort = {}
    cohort['train'] = casecontrol(cond, drug, train_id)
    cohort['valid'] = casecontrol(cond, drug, val_id)
    cohort['test'] = casecontrol(cond, drug, test_id)
    print('Done')
    
    for t, c in cohort.items():
        for k, udict in c.items():
            print(f'Constructing << cohort_{drug}_{cond}_{t}_{k} >>...')
            features = {}
            concept, age, segment, label = [], [], [], []
            for id in tqdm(udict.keys()):
                pg = person.loc[id]['gender_source_value']
                concept.append(
                    [pg] + list(udict[id]['code'].astype(str).values)[-1000:])
                age.append(
                    [udict[id]['age'].iloc[0]] + list(udict[id]['age'].values)[-1000:])
                segment.append(
                    [udict[id]['segment'].iloc[0]] + list(udict[id]['segment'].values)[-1000:])
                label.append(
                    [0] + list(udict[id]['label'].values)[-1000:])

            features['concept'] = concept
            features['age'] = age
            features['segment'] = segment
            features['label'] = label
            tokenized = tokenizer(features)
            picklesave(tokenized, f'../usedata/tokens/cohort_{drug}_{cond}_{t}_{k}_token_gender1.pkl')

def r2(v):
    return "{:.2f}".format(round(v, 2))
def r3(v):
    return "{:.3f}".format(round(v, 3))
def r4(v):
    return "{:.4f}".format(round(v, 4))

def opentext(path):
    all_lines = []
    with open(path, 'r') as f:
        file = f.readlines()
        for line in file:
            all_lines.append(line.strip())
    return all_lines

def findstring(string, substring):
    indices = []
    index = -1
    while True:
        index = string.find(substring, index + 1)
        if index == -1:
            break
        indices.append(index)
    
    return indices


def normalize_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    
    # min_val == max_val인 경우 처리 (모든 값이 동일한 경우)
    if min_val == max_val:
        return [1] * len(lst)
    
    # 정규화
    normalized_lst = [1 + (x - min_val) * (5 - 1) / (max_val - min_val) for x in lst]
    
    return normalized_lst


def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi / 2 or (angle >= np.pi * 3 / 2):
        alignment = "left"
    else: 
        alignment = "right"
        rotation = rotation + 180
    return rotation, alignment


def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 4
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor",
            fontsize=17
        ) 


def icdtoconcept(v):
    import re
    if bool(re.search(r'^I2[1-2]|I252', v)):
        return 'Myocardial infarction'
    elif bool(re.search(r'^(I42|I43|I50)|I099$|I110$|I130$|I132$|I255$', v)):
        return 'Heart failure'
    elif bool(re.search(r'^(I70|I71)|I731$|I738$|I739$|I771$|I790$|I792$|K551$|K558$|K559$|Z958$|Z959$', v)):
        return 'Peripheral vascular disease'
    elif bool(re.search(r'^(I60|I61|I62|I63|I64|I65|I66|I67|I68|I69)', v)):
        return 'Cerebrovascular disease'
    elif bool(re.search(r'^(F00|F01|F02|F03|G30)|F051$|G311$', v)):
        return 'Dementia'
    elif bool(re.search(r'^(J40|J41|J42|J43|J44|J45|J46|J47|J60|J61|J62|J63|J64|J65|J66|J67)|I278$|I279$|J684$|J701$|J703$', v)):
        return 'Pulmonary disease'
    elif bool(re.search(r'^(M05|M06|M32|M33|M34)|M315$|M351$|M353$|M360$', v)):
        return 'Connective tissue disease'
    elif bool(re.search(r'^(K25|K26|K27|K28)', v)):
        return 'Peptic ulcer disease'
    elif bool(re.search(r'^(B18|K73|K74)|K700$|K701$|K702$|K703$|K709$|K713$|K714$|K715$|K717$|K760$|K762$|K763$|K764$|K768$|K769$|Z944$', v)):
        return 'Mild liver disease'
    elif bool(re.search(r'I850$|I859$|I864$|I982$|K704$|K711$|K721$|K729$|K765$|K766$|K767$', v)):
        return 'Liver disease'
    elif bool(re.search(r'E10$|E11$|E12$|E13$|E14$|E100$|E101$|E106$|E108$|E109$|E110$|E111$|E116$|E118$|E119$|E120$|E121$|E126$|E128$|E129$|E130$|E131$|E136$|E138$|E139$|E140$|E141$|E146$|E148$|E149$', v)):
        return 'Uncomplicated diabetes'
    elif bool(re.search(r'E102$|E103$|E104$|E105$|E107$|E112$|E113$|E114$|E115$|E117$|E122$|E123$|E124$|E125$|E127$|E132$|E133$|E134$|E135$|E137$|E142$|E143$|E144$|E145$|E147$', v)):
        return 'Complicated diabetes'
    elif bool(re.search(r'^(G81|G82)|G041$|G114$|G801$|G802$|G830$|G831$|G832$|G833$|G834$|G839$', v)):
        return 'Paraplegia and hemiplegia'
    elif bool(re.search(r'N03[2-7]|N05[2-7]|Z49[0-2]|^(N18|N19)|I120$|I131$|N250$|Z940$|Z992$', v)):
        return 'Renal disease'
    elif bool(re.search(r'^C0[0-9]|^C1[0-9]|^C2[0-6]|^C3[0-4]|^C3[7-9]|^C4[0-1]|^C4[5-9]|^C5[0-8]|^C6[0-9]|^C7[0-6]|^C8[1-5]|^C9[0-7]|^(C43|C88)', v)):
        return 'Malignant tumor'
    elif bool(re.search(r'^(C77|C78|C79|C80)', v)):
        return 'Metastatic carcinoma'
    elif bool(re.search(r'^(B20|B21|B22|B24)', v)):
        return 'HIV'
    else:
        return 'None'

