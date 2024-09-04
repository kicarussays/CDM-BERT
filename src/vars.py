# from src.packages import *
# from src.utils import *

# stcm = pd.read_csv('../../ehr_file/stcm.csv')
# person = pd.read_csv('../../ehr_file/person.csv')
# person = person.set_index('person_id')

# stcm = stcm[stcm['source_name'].notnull()]
# stcm['source_name'] = stcm['source_name'].str.lower()
# stcm_d = stcm[(stcm['domain'] == 'Drug') & (stcm['vocabulary'].isin(['RxNorm', 'RxNorm Extension']))]
# stcm_m = stcm[(stcm['domain'] == 'Measurement') & (stcm['vocabulary'] == 'LOINC')]
# stcm_c = stcm[(stcm['domain'] == 'Condition') & (stcm['vocabulary'] == 'SNOMED')]
# stcm_p = stcm[(stcm['domain'] == 'Procedure') & (stcm['vocabulary'] == 'SNOMED')]

# KEYWORDS = {
#     'ICH': 'intrac.*hemo',
#     'Anticoagulant': 'rivaroxaban|apixaban|edoxaban|dabigatran|warfarin|aspirin|heparin|enoxaparine|dalteparin',
# }

# # ICH
# CIDS = {
#     'ICH': pd.unique(stcm_c[stcm_c['source_name'].str.contains(KEYWORDS['ICH'])]['omop_concept_id'])
# }
# DIDS = {
#     # noac + warfarin + aspirin + heparin + lmwh
#     'Anticoagulant': drug_id(pd.unique(stcm_d[stcm_d['source_name'].str.contains(KEYWORDS['Anticoagulant'])]['omop_concept_id']))
# }

