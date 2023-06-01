# Analyse sémantique AMR pour le français par transfert translingue 

Ce repo est l'implémentation de [Analyse sémantique AMR pour le français par transfert translingue (Kang et al., 2023)](https://coria-taln-2023.sciencesconf.org/456133/document). 

## 1. Installation 
Le code est testé en Python 3.9. Il est fortement recommendé d'utiliser conda pour gérer l'environment. 
Pour installer les packages nécesssaires, exécutez la commande suivante:  
```
pip install -r requirements.txt 
```
Importez le package AMR pour le prétraitement + évaluation + post-traitement de graphes AMR avec la commande suivante sur le root du project: 

```
git clone https://github.com/RikVN/AMR.git
```
Cela devrait créer `French_AMR_Parser/AMR` dans ce projet. 

## 2. Télécharger les données
- AMR 
  - AMR est disponible sur [LDC](https://catalog.ldc.upenn.edu/LDC2020T02) et il faut avoir une licence pour télécharger les données.
  - Pour les données d'évaluation en français (FR_LPP_GOLD, FR_SILVER), veuillez voir le page [data](data/README.md). 


- UCCA 
  - Pour télécharger les données, `./download_ucca_corpus.sh`


- Corpus Parallèle
    - Pour télécharger les données, `./download_parallel_corpus.sh`. Cela peut prendre plus que 30 minutes ⏳ 

  

## 3. Prétraitement


- Prétraitement d'AMR   
  - Pour faire prétraitement (linéarisation) d'AMR, nous utilisons le script de [Van Noord](https://github.com/RikVN/AMR). Voir le repo originel pour plus de détails. 
  - Après la linéarisation, les données doivent organisé comme suit: 
  ```Add tree
    - data
        - AMR
          - training
            - en
              - train.graph      # Structured AMR graph, delimited with a blank line
              - train.txt.sent   # Sentences corresponding to the AMR graphs 
              - train.txt.tf     # Linearized AMR graph 
          - dev
            - en
          - test
            - en
            - fr
            - de
            ...
  ```

- Prétraitement de UCCA
  - `./preprocess_ucca_corpus.sh`
  - Ce processus va créer des graphes UCCA linéarisés dans le dossier `data/UCCA`


-  Prétraitement du corpus parallèle
    - `./preprocess_parallel_corpus.sh`
    - Ce processus va filtrer des paires de phrases parallèlels selon leur longeuer, ratio entre les caractères alignés. 

## 4. Entraînement
  - `./run_training.sh` 
  - Pour voir plus sur les paramètres, consultez le sript `scripts/train_amr_parser.py`

## 5. Evaluation
  - `/.run_test.sh`
  - Pour voir plus sur les paramètres, consultez le sript `scripts/run_test.py`

## 6. Référence
```
Kang J., Coavoux M., Lopez C., Scwhab D. (2023) Analyse sémantique AMR pour le français par transfert translingue : 30e Conférence sur le Traitement Automatique des Langues Naturelles (TALN 2023)
```