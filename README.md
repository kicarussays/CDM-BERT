# CDM-BERT

Official code for "Pretrained Patient Trajectories for Adverse Drug Event Prediction Using Common Data Model-based Electronic Health Records".

## Abstract
**Background**. Pretraining electronic health record (EHR) data using language models by treating patient trajectories as natural language sentences has enhanced performance across various medical tasks. However, EHR pretraining models have never been utilized in adverse drug event (ADE) prediction. We constructed and externally validated the EHR pretraining model for several ADE prediction tasks and qualitatively analyzed the important features of each ADE cohort.


**Methods**. A retrospective study was conducted on observational medical outcomes partnership (OMOP)-common data model (CDM) based EHR data from two separate tertiary hospitals. The data included patient information in various domains such as diagnosis, prescription, measurement, and procedure. For pretraining, codes were randomly masked, and the model was trained to infer the masked tokens utilizing preceding and following history. In this process, we adopted domain embedding (DE) to provide information about the domain of the masked token, preventing the model from finding codes from irrelevant domains. For qualitative analysis, we identified important features using the attention matrix from each finetuned model.

**Results**. 510,879 and 419,505 adult inpatients from two separate tertiary hospitals were included in internal and external datasets. EHR pretraining model with DE outperformed all the other baselines in all cohorts. For feature importance analysis, we demonstrated that the results were consistent with priorly reported background clinical knowledge. In addition to cohort-level interpretation, patient-level interpretation was also available.

**Conclusions**. EHR pretraining model with DE is a proper model for various ADE prediction tasks. The results of the qualitative analysis were consistent with background clinical knowledge.

