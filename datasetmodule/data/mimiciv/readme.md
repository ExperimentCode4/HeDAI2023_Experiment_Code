## Readme
This file contains a description of the files contained within the mimiciv data folder.  
These files are the primary source of auxiliary data for the construction of 
relevant auxiliary hierarchies and other domain knoweldge that can be added to the EHR graph.

### used_loinc_lab_codes.csv
This file contains the loinc codes used for laboratory tests in the OMOP CDM mapped MIMIC-IV dataset.  
The codes are used to heavily limit the construction time of the axial hierarchy for loinc codes by subsampling
the relevant parts of the hierarchy.  
The codes were queried from a Google Big Query setup over the schema of the MIMIC-IV data as:
```sql
select distinct C.concept_code, C.vocabulary_id
from `mimiciv.cdm_measurement` M
join `mimiciv.voc_concept` C on M.measurement_concept_id = C.concept_id
where C.vocabulary_id = 'LOINC'
```

### loinc_full_hierarchy
This file contains the complete multiAxialHierarchy of LOINC codes downloadable from the [LOINC website](https://loinc.org/file-access/?download-id=470626).  
A free user is required to download the file. The hierarch is processed for removal of all codes not used in the MIMIC-IV data.


### used_proc_codes
This file contains the procedure codes used for MIMIC-IV.
The codes are used to heavily limit the construction time and embedding space of the ICD-9 procedure codes hierarchy 
by subsampling the relevant parts of the hierarchy and omitting parts not relevant.  
The codes were queries from a Google Big Query setup over the schema of the OMOP CDM mapped MIMIC-IV data as:
```sql
SELECT DISTINCT C.concept_code
FROM `mimiciv.cdm_procedure_occurrence` CO
join `mimiciv.voc_concept` C on C.concept_id = CO.procedure_concept_id
where C.vocabulary_id = 'ICD9Proc'
```

### proc_subset_hrchy
The hierarchy of child-parent codes forming the ICD-9 procedure code hierarchy. The file can be created using the `prepare_procedures_subset`
function from the `proc_utils.py` file. The only requirement is a file with the subset procedure codes used within
the MIMIC-IV database `used_proc_codes`.

### diag_full_hrchy
A json file with the full ICD9 diagnosis code hierarchy. The hierarchy contains all ICD9 diagnosis codes and their ancestry.

### used_diag_codes
Contains the subset of ICD9 diagnosis codes we are interested in, extracted from the MIMIC-IV dataset. the table `mimiciv.icd9_filtered` is not
originally part of the OMOP CDM mapping, but a table of manually constructed diagnosis codes where some diagnosis codes for infants and other non-discernable 
codes have been omittet.  
The codes were queried from a Google Big Query setup over the schema of the OMOP CDM mapped MIMIC-IV data as:
```sql
select distinct FILT.level4
from `mimiciv.icd9_filtered` as FILT
join `mimiciv.cdm_condition_occurrence` CO on CO.condition_source_value = FILT.level4
join `mimiciv.voc_concept` C on C.concept_id = CO.condition_source_concept_id
where C.vocabulary_id = 'ICD9CM'
group by FILT.level4
having count(*) > 500;
```

### used_atc_codes
Contain a subset of atc codes used in the conversion from RxNorm to atc.
The subset is extracted from the OMOP CDM vocabularies with the following query:
```sql
select distinct C3.concept_code as atc_code
from `mimiciv.cdm_drug_exposure` DE
join `mimiciv.voc_concept` C1 on C1.concept_id = DE.drug_concept_id
join `mimiciv.voc_concept_ancestor` CA on CA.descendant_concept_id = C1.concept_id
join `mimiciv.voc_concept` C2 on C2.concept_id = CA.ancestor_concept_id
join `mimiciv.voc_concept_relationship` CR on C2.concept_id = CR.concept_id_1
join `mimiciv.voc_concept` C3 on C3.concept_id = CR.concept_id_2
where DE.drug_concept_id > 0
and CA.min_levels_of_separation = 1
and C2.vocabulary_id = 'RxNorm'
and C2.concept_class_id = 'Clinical Drug Form'
and CR.relationship_id = 'RxNorm - ATC'
```