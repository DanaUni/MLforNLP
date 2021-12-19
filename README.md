# Portfolio code for the course Machine Learning for Natural Language Processing 2021

The code requires the following packages to run:
```
csv
typing
collections
sys
pandas
sklearn
argparse
gensim
```

## overview

The folder contains the following files and subfolder:
- prepocessing_2021.ipynb
- basic_evaluation.ipynb
- basic_system.ipynb
- ner_machine_learning.py
- feature_extraction.py
- feature_ablation.py
- /settings
  - conversions.tsv
  - generic_suffixes_toponomy_UK_IR.txt

The settings folder contains the conversions.tsv for converting the NE labels and the generic_suffixes_toponomy_UK_IR.txt file for checking the token suffixes.

The code can be run as follows.

### prepocessing_2021.ipynb
1. Open the preprocessing_conll_2021.ipynb file using Jupyter Lab or Jupyter Notebook.
2. If necessary, change the paths to the conll input files in the cells under the headers "Change paths" near the end of the notebook.
3. If necessary, change the column identifier to that of the NE label column in the input files.
4. Run the whole notebook at once or run the individual cells from top to bottom.
5. Running this notebook has to be repeated for different input files (SpaCy, Stanford, conll2003.train, conll2003.dev and conll2003.test)

### basic_evaluation.ipynb
1. Open the basic_evaluation.ipynb file using Jupyter Lab or Jupyter Notebook.
2. If necessary, change the paths to the conll inputfiles for the gold labels and the system labels in the cell under the header "Change paths" near the end of the notebook.
3. If necessary, change the column identifiers to that of the NE label columns in the input files.
4. If necessary, change the system name to the name of model used in the variable system_name.
5. Run the whole notebook at once or run the individual cells from top to bottom.
6. Running this notebook has to be repeated for evaluations on different files (mainly for SpaCy and Stanford, as the feature_ablation.py file is used for the more advanced systems)

### basic_system.ipynb
(used only to create the basic logreg model system with only the token as feature)
1. Open the basic_system.ipynb file using Jupyter Lab or Jupyter Notebook.
2. If necessary, change the paths to the conll trainingfile, inputfile and outputfile in the cell under the header "Change paths" near the end of the notebook.
3. Run the whole notebook at once or run the individual cells from top to bottom.

### extract_features.py
1. Change directory in the commandline to the current directory (Portfolio_code_-_dwk320).
2. This file can be run from the command line using 3 positional arguments and 1 keyword arguments:

  required:
  - argument 0: name of the python program
  - argument 1: path to the inputfile
  - argument 2: path to the outputfile

  optional:
  - argument 3: path to the toponomyfile  (the default path is that of the file included in the submission folder, no need to change the path)

  for example:

    python feature_extraction.py "..\data\conll2003.train-preprocessed.conll" "..\data\conll2003.train-preprocessed-added_features.conll"

### ner_machine_learning.py
1. Change directory in the commandline to the current directory (Portfolio_code_-_dwk320) if not already there.
2. This file can be run from the command line using 5 positional arguments and 4 keyword arguments:

  required:
  - argument 0: name of the python program
  - argument 1: path to the trainingfile
  - argument 2: path to the inputfile
  - argument 3: path to the outputfile
  - argument 4: identifier of the gold NE labels column

  optional:
  - argument 5: -mod, --models, which models to use ("logreg", "NB" or "SVM")
  - argument 6: -sfeat, --selected_features, which features to use ("token", "POS", "chunk", "caps", "digits" and/or "place_suffix")
  - argument 7: -use, --use_embeddings, if you want to use the word_embeddings ("True"/"False") (if "True", include path to word embedding model in arg 5)
  - argument 8: -lmp, --language_model_path, the path to the language model (word embedding model)

  for example:

    python ner_machine_learning.py "..\data\conll2003.train-preprocessed-added_features.conll" "..\data\conll2003.dev-preprocessed-added_features.conll" "..\data\conll2003.dev-preprocessed-added_features_token-POS.conll" "3" -mod "logreg" -sfeat "token" "POS"

### feauture_ablation.py
1. Change directory in the commandline to the current directory (Portfolio_code_-_dwk320) if not already there.
2. This file can be run from the command line using 5 positional arguments and 6 keyword arguments:

  required:
  - argument 0: name of the python program
  - argument 1: path to the trainingfile
  - argument 2: path to the inputfile
  - argument 3: path to the outputfile
  - argument 4: identifier of the gold NE labels column

  optional:
  - argument 5: -mod, --models, which models to use ("logreg", "NB" or "SVM")
  - argument 6: -feat, --features, which features to use ("token", "POS", "chunk", "caps", "digits" and/or "place_suffix")
  - argument 7: -pl, --print_latex, to indicate if the table should be printed in latex format ("True"/"False")
  - argument 8: -use, --use_embeddings, if you want to use the word_embeddings ("True"/"False") (if "True", include path to word embedding model in arg 5)
  - argument 9: -lmp, --language_model_path, the path to the language model (word embedding model)
  - argument 10: -exo, --exclude_O_class, to indicate if the O class should be excluded from the tables ("True"/"False") (for the report)

  for example:

    python feature_ablation.py "..\data\conll2003.train-preprocessed-added_features.conll" "..\data\conll2003.dev-preprocessed-added_features.conll" "..\data\conll2003.dev-preprocessed-added_features_token-digits.conll" "3" -mod "SVM" -feat "token" "digits" -exo "True"
