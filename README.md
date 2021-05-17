# Infection Mechanism and Spectrum Prediction (IMSP) Model

Hangyu Du*, Feng Chen*, Hongfu Liu, and Pengyu Hong "[Network-based Virus-Host Interaction Prediction with Application to SARS-CoV-2](https://www.cell.com/patterns/fulltext/S2666-3899(21)00062-3?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389921000623%3Fshowall%3Dtrue)", Cell Patterns Volumn 2, Issue 5. (* indicates equal contribution)

SARS-CoV-2, a novel single-stranded RNA coronavirus causing COVID-19, is mounting an unprecedented threat against our society and the world. Although tremendous efforts have been devoted into SARS-CoV-2 research, most of them either focused on a few proteins or only provided high-level overviews. Deeper and more comprehensive analyses are needed to shed new light onto the molecular mechanisms underlying the COVID-19 pandemic. Moreover, there is a massive amount of data and knowledge about highly relevant RNA viruses which have yet to be fully utilized. In this work, we constructed a multi-layer virus-host interaction network to incorporate these data and knowledge. We developed a machine-learning-based method to predict virus-host interactions at both protein and organism levels. Our approach revealed five potential infection targets of SARS-CoV-2 and 19 highly possible interactions between SARS-CoV-2 proteins and human proteins in the innate immune pathway.


## Reference
If you find IMSP is useful for your research, please consider citing the following papers:

```bash

@article{DU2021100242,
    title = {Network-based virus-host interaction prediction with application to SARS-CoV-2},
    journal = {Patterns},
    volume = {2},
    number = {5},
    pages = {100242},
    year = {2021},
    issn = {2666-3899},
    doi = {https://doi.org/10.1016/j.patter.2021.100242},
    url = {https://www.sciencedirect.com/science/article/pii/S2666389921000623},
    author = {Hangyu Du and Feng Chen and Hongfu Liu and Pengyu Hong},
    keywords = {coronavirus, COVID-19, SARS-CoV-2, machine learning, interaction prediction, protein-protein interaction, virus-host interaction network},
}
```

## Installation

- Clone the repository
- Run the following command
    ```bash
    pip install -r requirements.txt
    ```
- If the installation of spaCy gives an error, try this
    ```bash
    python -m spacy download en_core_web_sm
    ```
## Usage
- ### Arguments:
  --bg: bool, if set to True, the model will only build the network in NetworkX's GML format. Default: False

  --eval: bool, if set to True, the model will only run the evaluation part without making predictions. Default:
  False

  --eval_iter: int, the number of full cross-validation iterations for performance evaluation. Default: 30

  --leave_one_out: bool, if set to True, will perform leave_one_out experiment on infected hosts. Default: False

- ### Example
     ```python
    '''Perform link prediction task'''
    python main.py
    ```

     ```python
    '''Perform a 50-run model performance measurement task'''
    python main.py --eval True --eval_iter 50
    ```

    ```python
    '''Only constructs the network only w/o performing any downstream tasks, i.e. link prediction/performance measurement'''
    python main.py --bg True
    ```
  
     ```python
    '''Perform link prediction task'''
    python main.py --leave_one_out True
    ```

- ### Input:
    - Pair-wise similarity matrices. You can obtain pair-wise similarity matrices for protein homologs by using
      our [Data Parser](https://github.com/hangyu98/IMSP-Parser).
    - Protein-Protein Interactions (PPI) and infection relationships should be collected and feed into
      /network/network_data.py following this formatting:
      ```python
      {
          'group_1': 'virus protein',
          'type_1': ['nsp15'],
          'host_list_1': ['Severe acute respiratory syndrome coronavirus 2'],
          'group_2': 'host protein',
          'type_2': ['IRF3', 'RIG-I'],
          'host_list_2': ['Homo sapiens', 'Felis catus', 'Macaca mulatta', 'Canis lupus familiaris',
                          'Rhinolophus ferrumequinum', 'Mesocricetus auratus'],
          'relation': 'interacts'
      }
      ```
      Specifically, the input above will connect nsp15 of SARS-CoV-2 to IRF3 and RIG-I in the following hosts: Homo
      sapiens, Felis catus, Macaca mulatta, Canis lupus familiaris, Rhinolophus ferrumequinum, and Mesocricetus auratus.
      The link type will be set as "interacts" (PPI).
    - Protein functions should be collected and feed into /network/network_data.py following this formatting:
      ```python
      protein_function_data = {
          'ACE2': 'virus receptor activity',
          'IRF3': 'interferon regulatory factor',
          ...
      }
      ```
    - In addition to these, please also provide lists of the full name for viruses and hosts in /network/network_data.py as follows 
      ```      
      # list of hosts
      list_of_hosts = ['Homo sapiens', 'Felis catus', 'Mus musculus',
                 'Rattus norvegicus', 'Canis lupus familiaris',
                 'Ictidomys tridecemlineatus', 'Camelus dromedarius', 'Bos taurus', 'Pan troglodytes',
                 'Gallus gallus', 'Oryctolagus cuniculus', 'Equus caballus', 'Macaca mulatta', 'Ovis aries',
                 'Sus scrofa domesticus', 'Rhinolophus ferrumequinum', 'Mesocricetus auratus']
      # list of viruses
      list_of_viruses = ['Human coronavirus OC43', 'Human coronavirus HKU1',
                   'Middle East respiratory syndrome-related coronavirus',
                   'Severe acute respiratory syndrome coronavirus 2',
                   'Severe acute respiratory syndrome-related coronavirus', 'Human coronavirus 229E',
                   'Human coronavirus NL63']
      ```
- ### Output:
    - The link prediction results are available at ```/data/prediction```
        - ```/data/prediction/prediction_infects.csv``` contains infection predictions
        - ```/data/prediction/prediction_interacts.csv``` contains PPI predictions
    - The performance evaluation results are available at ```/data/evaluation```
        - ```/data/prediction/performance_summary.csv``` contains the means and SDs for all evaluation metrics
        - ```/data/prediction/performance_details.csv``` logs the performance for all the models in each run

- ### Customizable filter:
    - We understand that the prediction results might need customized filters to validate in different scenarios. We
      have provided some sample code at ```/customize/customized_filter.py```

## Jupyter Notebook
- Usage: put all three files (main.ipynb, cytoStyle.json, predictStyle.json) into the home directory of the project. 
- Link: https://drive.google.com/drive/folders/18ZUb5NGaUJHZuCj7uo-DA_CzCEkpeh9p?usp=sharing
