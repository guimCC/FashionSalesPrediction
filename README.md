# FashionSalesPrediction
### Predicció de les tendències de vendes d’articles de moda sobre un dataset multimodal amb mètodes d'aprenentatge automàtic tradicionals

Aquest projecte explora la viabilitat de predir les tendències de vendes en la indústria de la moda utilitzant tècniques tradicionals d'aprenentatge automàtic. Amb dades provinents del dataset multimodal **VISUALLE**, el projecte analitza la relació entre les característiques d'articles de moda i les seves vendes. El dataset inclou informació visual, metadades textuals, dades de tendències de Google i registres de vendes de més de 5000 articles de roba.

L'objectiu principal és determinar si tècniques d'aprenentatge automàtic més simples poden substituir enfocaments més complexos, com el deep learning, per a tasques de predicció en dades multimodals. El projecte comença amb un enfocament de regressió per predir les vendes totals, però canvia a una classificació binària per identificar articles amb vendes destacades a causa de la distribució desbalancejada de les dades.

Les estratègies implementades inclouen:

- Enginyeria de dades per simplificar i fer més interpretables les característiques, com ara les dades visuals i les tendències de Google.
- Aplicació de tècniques de balanceig de classes com SMOTE i submostreig per abordar el desbalanceig de les dades.
- Avaluació de diversos models d'aprenentatge automàtic com la regressió lineal, Random Forest i XGBoost.

Els resultats mostren que, tot i certs avenços, les tècniques tradicionals tenen limitacions importants per capturar la complexitat de les dades multimodals. Això destaca la necessitat d'explorar enfocaments més avançats, com l'ús de xarxes neuronals i arquitectures basades en transformers, especialment per a integrar millor les dades visuals amb altres característiques.

Aquest projecte posa en evidència els reptes de treballar amb dades desbalancejades i multimodals en la predicció de tendències de vendes. Alhora, proporciona una base sòlida per a futures investigacions en aquest camp, amb la idea de millorar la precisió i l'eficàcia dels models predictius per a la indústria de la moda.

### Estructura del Repositori i d'execució
- Dataset:
  - Descarregar i extreure el **Dataset** a partir del següent enllaç: [VISUELLE](https://paperswithcode.com/dataset/visuelle)
  - **NOTA**: a les notebooks següents s'ha de canviar la variable _path_ per el lloc d'extracció del dataset. S'indica als fitxers.
- Notebooks:
  - [create_CLIP_embeddings.ipynb](clean_notebooks/create_CLIP_embeddings.ipynb): S'ha de correr aquest script primer per a generar els _embeddings_.
  - [first_approach_BoVW.ipynb](clean_notebooks/first_approach_BoVW.ipynb): Aquest fitxer inclou el codi per a realitzar les features del _BagOfVisualWords_. 
  - [model_notebook.ipynb](clean_notebooks/model_notebook.ipynb): Aquesta és la notebook principal, on s'an d'incloure els fitxers generats amb les notebooks anteriors i el dataset, tal com està inclòs dins la notebook.
  - **NOTA**: No estan totes les versions i iteracions que s'han mencionat a l'informe, si no només la majoria de versions finals.
