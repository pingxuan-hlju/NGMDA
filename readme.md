### NGMDA

***

**Inferring drug-associated microbes with knowledge from neighborhoods and whole graph.**

### Operating environment

***

- python 3.8.15

- pytorch 1.9.0

- dgl 0.9.0

- dgl-cuda10.2

- numpy 1.23.5

### Data description

***

- net1: known adjacent matrix for drugs and microbes, i.2., interaction.
- microbe_features: contains feature vectors of 173 microbes.
- drug_names: contains names of 1373 drugs.
- microbe_names: contains names of 173 microbes.
- drugsimilarity: similarity between drugs.
- micr_cosine_sim: similarity between microbes computed by cosine similarity.
- model: defines NGMDA.
- tools: contains earlying stop function.
- data_loader: a class for loading data.
- fold0-fold5: evaluation file for further analysis.
- case_analysis: contains some files for case analysis.

### Run steps

***

1. install operating environment.
2. run main.py.

note, To simulate the real situation, we use balanced sample for training and unbalance smaple for test.

after running, you can get acc, auc and aupr of our model in each folder.

