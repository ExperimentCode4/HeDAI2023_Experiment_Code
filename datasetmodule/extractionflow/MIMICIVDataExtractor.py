from datasetmodule.extractionflow.DataExtractor import DataExtractor


class MimicIVDataExtractor(DataExtractor):
    def __init__(
            self,
            args
    ):
        super(MimicIVDataExtractor, self).__init__(args)

    def extract_entities(self, conf):
        if self.ent_extract:
            for entity in conf['entity_extract']:
                func = getattr(self, entity['query'])
                self.extract(func, entity['query'])
        else:
            print(f'ent_extract set to 0, using local parquet files')

    def extract_relations(self, conf):
        if self.rel_extract:
            for relation in conf['relation_extract']:
                func = getattr(self, relation['query'])
                self.extract(func,  relation['sub'], relation['obj'])
        else:
            print(f'rel_extract set to 0, using local parquet files')

    def extract_features(self, conf):
        if self.feat_extract:
            for feature in conf['feature_extract']:
                func = getattr(self, feature['query'])
                self.extract(func)
        else:
            print(f'feat_extract set to 0, using local parquet files')

    # ----------------- Entities -------------------
    def person(self, sub):
        return f"""select distinct P.person_id as {sub}
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_condition_occurrence` CO
                   join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_person` P on P.person_id = CO.person_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.condition_source_concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.ccs_single` CCS on CCS.icd9 = CO.condition_source_value
                   join `tomer-personal.mimiciv_cdm_2021_07_30.icd9_filtered` FILT on FILT.level4 = CO.condition_source_value
                   where C.vocabulary_id = 'ICD9CM'
                   and CO.condition_source_value in (
                       select distinct FILT.level4 from `tomer-personal.mimiciv_cdm_2021_07_30.icd9_filtered` as FILT
                       join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_condition_occurrence` CO on CO.condition_source_value = FILT.level4
                       join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.condition_source_concept_id
                       where C.vocabulary_id = 'ICD9CM'
                       group by FILT.level4
                       having count(*) > 500);"""

    def diagnosis(self, sub):
        return f"""select distinct FILT.level4 as {sub} 
                   from `tomer-personal.mimiciv_cdm_2021_07_30.icd9_filtered` as FILT
                   join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_condition_occurrence` CO on CO.condition_source_value = FILT.level4
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.condition_source_concept_id
                   where C.vocabulary_id = 'ICD9CM'
                   group by FILT.level4
                   having count(*) > 500;"""

    def medication(self, sub):
        return f"""select distinct C2.concept_code as {sub}
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_drug_exposure` DE
                   join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_person` P on P.person_id = DE.person_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C1 on C1.concept_id = DE.drug_concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept_ancestor` CA on CA.descendant_concept_id = C1.concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C2 on C2.concept_id = CA.ancestor_concept_id
                   where DE.drug_concept_id > 0
                   and CA.min_levels_of_separation = 1
                   and C2.vocabulary_id = 'RxNorm'
                   and C2.concept_class_id = 'Clinical Drug Form'"""

    def labtest(self, sub):
        return f"""select distinct C.concept_code || '-N' as {sub}
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_measurement` M
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on M.measurement_concept_id = C.concept_id
                   where C.vocabulary_id = 'LOINC'
                   union all (
                       select distinct C.concept_code || '-A'
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_measurement` M
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on M.measurement_concept_id = C.concept_id
                   where C.vocabulary_id = 'LOINC')"""

    def procedure(self, sub):
        return f"""SELECT DISTINCT C.concept_code as {sub}
                   FROM `tomer-personal.mimiciv_cdm_2021_07_30.cdm_procedure_occurrence` CO
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.procedure_concept_id
                   where C.vocabulary_id = 'ICD9Proc'"""

    # ----------------- Relations -------------------

    def person_diagnosis(self, sub, obj):
        return f"""select distinct CO.person_id as {sub}, CO.condition_source_value as {obj}
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_condition_occurrence` CO
                   join `tomer-personal.mimiciv_cdm_2021_07_30.icd9_filtered` FILT on FILT.level4 = CO.condition_source_value
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.condition_source_concept_id
                   where C.vocabulary_id = 'ICD9CM'
                   and CO.condition_source_value in (
                       select distinct FILT.level4 from `tomer-personal.mimiciv_cdm_2021_07_30.icd9_filtered` as FILT
                       join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_condition_occurrence` CO on CO.condition_source_value = FILT.level4
                       join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.condition_source_concept_id
                       where C.vocabulary_id = 'ICD9CM'
                       group by FILT.level4
                       having count(*) > 500)"""

    def person_medication(self, sub, obj):
        return f"""select distinct de.person_id as {sub}, C2.concept_code as {obj}
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_drug_exposure` DE
                   join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_person` P on P.person_id = DE.person_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C1 on C1.concept_id = DE.drug_concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept_ancestor` CA on CA.descendant_concept_id = C1.concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C2 on C2.concept_id = CA.ancestor_concept_id
                   where DE.drug_concept_id > 0
                   and CA.min_levels_of_separation = 1
                   and C2.vocabulary_id = 'RxNorm'
                   and C2.concept_class_id = 'Clinical Drug Form'"""

    def person_labtest(self, sub, obj):
        return f"""select distinct M.person_id as {sub}, CASE when L.flag is NULL then C.concept_code || '-N' else C.concept_code || '-A' end as {obj} 
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_measurement` M
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on M.measurement_concept_id = C.concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.src_labevents` L on M.trace_id = L.trace_id"""

    def person_procedure(self, sub, obj):
        return f"""SELECT DISTINCT CO.person_id as {sub}, C.concept_code as {obj}
                   FROM `tomer-personal.mimiciv_cdm_2021_07_30.cdm_procedure_occurrence` CO
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.procedure_concept_id
                   where C.vocabulary_id = 'ICD9Proc'"""

    # ----------------- Features -----------------

    def person_features(self):
        return f"""select CP.person_id as person, CP.gender_concept_id as gender, CP.race_concept_id as race, CP.ethnicity_concept_id as ethnicity, SP.anchor_age as age 
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_person` CP
                   join `tomer-personal.mimiciv_cdm_2021_07_30.src_patients` SP on CAST(SP.subject_id as string) = CP.person_source_value"""

    def atc_code_features(self):
        return f"""select distinct C2.concept_code as medication, C3.concept_code as atc_code
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_drug_exposure` DE
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C1 on C1.concept_id = DE.drug_concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept_ancestor` CA on CA.descendant_concept_id = C1.concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C2 on C2.concept_id = CA.ancestor_concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept_relationship` CR on C2.concept_id = CR.concept_id_1
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C3 on C3.concept_id = CR.concept_id_2
                   where DE.drug_concept_id > 0
                   and CA.min_levels_of_separation = 1
                   and C2.vocabulary_id = 'RxNorm'
                   and C2.concept_class_id = 'Clinical Drug Form'
                   and CR.relationship_id = 'RxNorm - ATC'"""

    # ----------------- Others -------------------

    def ccs_old(self, sub, obj):
        return f"""select distinct P.person_id as {sub}, CCS.ccs as {obj} 
                   from `tomer-personal.mimiciv_cdm_2021_07_30.cdm_condition_occurrence` CO
                   join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_person` P on P.person_id = CO.person_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.condition_source_concept_id
                   join `tomer-personal.mimiciv_cdm_2021_07_30.ccs_single` CCS on CCS.icd9 = CO.condition_source_value
                   join `tomer-personal.mimiciv_cdm_2021_07_30.icd9_filtered` FILT on FILT.level4 = CO.condition_source_value
                   where C.vocabulary_id = 'ICD9CM'
                   and CO.condition_source_value in (
                       select distinct FILT.level4 from `tomer-personal.mimiciv_cdm_2021_07_30.icd9_filtered` as FILT
                       join `tomer-personal.mimiciv_cdm_2021_07_30.cdm_condition_occurrence` CO on CO.condition_source_value = FILT.level4
                       join `tomer-personal.mimiciv_cdm_2021_07_30.voc_concept` C on C.concept_id = CO.condition_source_concept_id
                       where C.vocabulary_id = 'ICD9CM'
                       group by FILT.level4
                       having count(*) > 500);"""

    def microbiology_old(self):
        return """SELECT DISTINCT m.subject_id, m.microevent_id, m.spec_type_desc, m.test_name, 
                  m.org_name, m.ab_name, m.interpretation  from mimiciv.microbiologyevents m"""

    def procedure_old(self):
        return """SELECT DISTINCT subject_id, icd_code as proc_code FROM mimiciv.procedures_icd"""
