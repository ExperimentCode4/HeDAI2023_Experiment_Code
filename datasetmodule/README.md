# Dataset Module
This file explains the usage of the datasetmodule for creating and formatting EHR graphs from a MIMIC-IV OMOP mapped dataset on Google Big Query.

## Prerequisites
### Auxiliary hierarchies
The project comes with a lot of preprocessed files to make the graph creation process easier. All auxiliary hierarchical
taxonomies are precomputed and placed in the `datasetmodule/data` folder. For more information on the preprocessed
files see the `README` in the `data` folder.

### MIMIC-IV dataset
MIMIC-IV is an extensive, freely available database consisting of deidentified health-related data for more than 380K 
patient visits from the critical care unit of the Beth Israel Deaconess Medical Center between 2008 and 2019. To generate
EHR graphs from this dataset, you first need access to the dataset using the following steps:

- Create an account at [physionet](https://physionet.org/login/?next=/settings/credentialing/)
- Complete relevant CITI [training](https://physionet.org/settings/training/)
- Sign the data use agreement ([DUA](https://physionet.org/content/mimiciv/view-dua/2.0/)).
- The full dataset can then be [accessed](https://physionet.org/content/mimiciv/2.0/)

### OMOP CDM
After obtaining the dataset, convert the dataset into the OMOP CDM format and setup a Google Big Query version of the
dataset using [THIS](https://github.com/OHDSI/MIMIC) Extract Transform Load workflow from the OHDSI community. Now would
be a good time to familiarize yourselves with the file located in the `datasetmodule/data` folder.  
Upload the icd9_filtered.csv table to your GBQ resource as a new table `icd9_filtered` with three columns `level4`, `level3`
and `level2`.

### Google Big Query credientials
Once you have a running instance of Google Big Query with the OMOP CDM mapped MIMIC-IV dataset, create a service user
for the dataset following [THIS](https://cloud.google.com/bigquery/docs/authentication) guide, and add the service
account key file to the `datasetmodule/conf` folder.  
Also insert the Google Big Query project id into the `datasetmodule/configuration.ini` file. (Remove .example extension
or create a new file).  
Your GBQ mimiciv dataset will be given a specific name such as `mimiciv_cdm_2021_07_30`. Insert this into the 
`datasetmodule/configuration.ini` file.

### Setup Neptune Logging
If you with to use neptune for logging experiments, setup a free neptune account [HERE](https://app.neptune.ai/). Create
a new neptune project and insert the project id into the configuration file at `datasetmodule/config.ini` (remove .example
extension or create a new file). Also insert the neptune projects api_token into the .env file (remove.example 
extension or create a new file).

### Experimental setup
Configuration of the graph creation process happens through the configuration file located at `datasetmodule/conf/mimiciv.json`.
The following fields do not need to be altered to run the general experiments:
- entity_extract: The entities to extract from mimic-iv 
- relation_extract: The relation to extract from mimic-iv
- feature_extract: The features to extract from mimic-iv
- entities: The entities that should be part of the graph
- relations: The relations that should be included in the graph
- aux_features: The features that should be added as graph embeddings

The following fields of "general" changes how the graph is created:
- diag_level: Specifies the aggregation level of patient diagnosis codes over the ICD-9 hierarchy. This parameter can be changed
to experiment with other aggregation levels 1 thorough 5 for varying complexities of the diagnosis prediction task.
- aux_features: Specifies if auxiliary domain hierarchies should be used to pre-initialize node features or not.
- plot_aux_feats: Used to generate the embedding monotonicity plot.

### Command Line Arguments
The script `dataset_main` takes three arguments:
-e: For extracting entities from the GBQ dataset and saving to parquet files
-r: For extracting relations from the GBQ dataset and saving to parquet files
-f: For extracting features from the BGQ dataset and saving to parquet files

The first run requires all parameters to be present, since no parquet files have been stored yet `python dataset_main.py -e -r -f`. Subsequent
runs can omit these arguments since all data has been saved to local parquet files (This saves a lot of time and querying).

### Graphlets
It is possible to extract and use graphlets for initializing node features. However, this requires a bit of work:
- Have the base system up and running with all entities relations and features extracted to parquet files.
- Run the script: `datasetmodeule/utils/graphlet.py` for extracting a list of edge counts from the extracted graph.
- The script will produce a file `sparse.csv` located in the `output` folder.
- Download the graphlet extraction software [PGD](https://github.com/nkahmed/PGD) from the paper `Efficient Graphlet Counting for Large Networks`.
- Copy the `sparse.csv` file to the root folder of PGD. 
- Run the software with `./pgd -f sparse.csv --micro sparse.micro` (This could take several days).
- Once done copy the result file `sparse.csv` from the PGD folder to the `datasetmodule/data` folder.
- Edit the `conf/mimiciv.json` aux_features configuration to be: 
```json 
        "aux_features": [
                {"data_file": "", "entity": "all", "function": "graphlets"}
        ]
```
- Run the function `convert_sparse_to_parquet` from the graphlet script
- Run the `dataset_main` script again.
- A graph is now generates with nodes preinitialized with graphlet features.