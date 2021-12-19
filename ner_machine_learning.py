from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import sys
import csv
import gensim
import argparse

feature_column_identifiers = {'token': 0, 'POS': 1, 'chunk': 2, 'caps': 4, 'hyphen': 5, 'digits': 6, 'place_suffix': 7}

def extract_embeddings_as_features_and_gold(conllfile, word_embedding_model):
    """
    Function that extracts features and gold labels using word embeddings

    :param conllfile: path to conll file
    :param word_embedding_model: a pretrained word embedding model
    :type conllfile: string
    :type word_embedding_model: gensim.models.keyedvectors.Word2VecKeyedVectors

    :return features: list of vector representation of tokens
    :return labels: list of gold labels
    """
    ### This code was partially inspired by code included in the HLT course, obtained from https://github.com/cltl/ma-hlt-labs/, accessed in May 2020.
    labels = []
    features = []
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t',quotechar='|')
    for row in csvreader:
        #check for cases where empty lines mark sentence boundaries (which some conll files do).
        if len(row) > 0:
            if row[0] in word_embedding_model:
                vector = word_embedding_model[row[0]]
            else:
                vector = [0]*300
            features.append(vector)
            labels.append(row[gold_column_identifier])
    return features, labels


def extract_features_and_labels(trainingfile):
    """
    Extracts labels and features from the trainingfile

    :param trainingfile: path to the file with the training data

    :returns: a list with a dictionary with the features for each row
    """
    data = []
    targets = []
    # TIP: recall that you can find information on how to integrate features here:
    # https://scikit-learn.org/stable/modules/feature_extraction.html
    with open(trainingfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                # Adjusted from sample_code_features_ablation_analysis.ipynb december 2021
                feature_dict = dict()
                for feature_name in selected_features:
                    row_index = feature_column_identifiers[feature_name]
                    feature_dict[feature_name] = components[row_index]
                ###
                data.append(feature_dict)

                targets.append(components[gold_column_identifier])
    return data, targets
    
def extract_features(inputfile):
    """
    Extracts features from the inputfile

    :param inputfile: path to the inputfile

    :returns: a list with a dictionary for each row containing the features for that row
    """
    data = []
    with open(inputfile, 'r', encoding='utf8') as infile:
        for line in infile:
            components = line.rstrip('\n').split()
            if len(components) > 0:
                # Adjusted from sample_code_features_ablation_analysis.ipynb december 2021
                feature_dict = {}
                for feature_name in selected_features:
                    row_index = feature_column_identifiers.get(feature_name)
                    feature_dict[feature_name] = components[row_index]
                ###
                data.append(feature_dict)

    return data
    
def create_classifier(train_features, train_targets, modelname):
    """
    Creates a classifier with the kind depending on the modelname, one-hot encodes the train_features and uses these to fit the model

    :param train_features: a list of dictionaries containing the training features
    :param train_targets: a list of the target labels
    :param modelname: a string to specify the kind of model

    :returns: the fitted model and the DictVectorizer vector
    """
    if modelname ==  'logreg':
        # TIP: you may need to solve this: https://stackoverflow.com/questions/61814494/what-is-this-warning-convergencewarning-lbfgs-failed-to-converge-status-1
        model = LogisticRegression(max_iter=500)
    elif modelname == 'NB':
        if use_embeddings:
            model = GaussianNB()
        else:
            model = BernoulliNB()
    elif modelname == 'SVM':
        model = svm.LinearSVC() #for the sake of runtime the linear kernel is used for both the one-hot encodings as well as embeddings


    vec = DictVectorizer()

    # If embeddings are used the DictVectorizer does not need to be used.
    if use_embeddings:
        features_vectorized = train_features
    else:
        features_vectorized = vec.fit_transform(train_features)

    model.fit(features_vectorized, train_targets)

    return model, vec
    
    
def classify_data(model, vec, inputdata, outputfile):
    """
    Extracts the features from the inputdata, one-hot encodes them, then uses those to predict the labels
    with the model and write the assigned classes to the outputfile

    :param model: fitted model
    :param vec: vectoriser to vectorise the features
    :param inputdata: path to the inputfile
    :param outputfile: path to the outputfile
    """

    if use_embeddings:
        features, labels = extract_embeddings_as_features_and_gold(inputdata, language_model)
    else:
        features = extract_features(inputdata)
        features = vec.transform(features)

    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for line in open(inputdata, 'r'):
        if len(line.rstrip('\n').split()) > 0:
            outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
            counter += 1
    outfile.close()

def split_arg_string_list(string_list):
    """
    Splits a string from an argument in the commandline back into a list when it is a string containing the delimiter comma.

    :param string_list: a list possibly containing a string format comma separated list

    :returns: the list that was inputted as a string
    """
    if "," in string_list[0]:
        return string_list[0].split(",")
    else:
        return string_list

def string_to_bool(bool_string):
    """
    Returns a boolean based on the string.

    :param bool_string: a string format of a boolean

    :returns: a boolean
    """
    if bool_string.lower() == "true":
        return True
    else:
        return False


def main(argv=None):
    """
    Runs the function to extract the training features and gold labels and use these to create a classifier and classify the data

    :param argv: a list of arguments to indicate the following -> argv[0]: python program used,
    argv[1]: path to the trainingfile, argv[2]: path to the inputfile, argv[3]: path to the outputfile,
    argv[4]: column_identifier of the gold NE labels, argv[5]: model, argv[6]: features to select,
    argv[7]: if embeddings should be used for the token, argv[8]: path to language model
    """
    global gold_column_identifier
    global selected_features
    global use_embeddings
    global language_model

    # Parser arguments
    parser = argparse.ArgumentParser(description='run ner_machine_learning.py')
    parser.add_argument('trainingfile', type=str)
    parser.add_argument('inputfile', type=str)
    parser.add_argument('outputfile', type=str)
    parser.add_argument('gold_column_identifier', type=str)
    parser.add_argument('-mod', '--models', type=str, nargs="*", default=['logreg'])
    parser.add_argument('-sfeat', '--selected_features', type=str, nargs="*", default=["token"])
    parser.add_argument('-use', '--use_embeddings', type=str, default="False")
    parser.add_argument('-lmp', '--language_model_path', type=str, default="..\models\GoogleNews-vectors-negative300.bin.gz")

    # Determine what input argv is provided.
    if argv is None:
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args(argv)

    # Argument variables
    trainingfile = args.trainingfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    gold_column_identifier = int(args.gold_column_identifier)
    models = args.models
    selected_features = args.selected_features
    language_model_path = args.language_model_path
    use_embeddings = string_to_bool(args.use_embeddings)

    # When models and features are input from the feature_ablation.py the models are a list in string format, separated by a comma.
    # The models and selected_features are replaced by their lists in readible list format.
    models = split_arg_string_list(models)
    selected_features = split_arg_string_list(selected_features)

    # If embeddings are used, only the token feature wil be used (no mixed representations)
    if use_embeddings:
        selected_features = ["token"]

    print("\nSelected model(s):", models)
    print("Selected feature(s):", selected_features)

    if use_embeddings:
        print("Loading language model...")
        language_model = gensim.models.KeyedVectors.load_word2vec_format(language_model_path, binary=True)
        training_features, gold_labels = extract_embeddings_as_features_and_gold(trainingfile, language_model)
    else:
        training_features, gold_labels = extract_features_and_labels(trainingfile)

    for modelname in models:
        print(f"Classifying with {modelname}...")
        ml_model, vec = create_classifier(training_features, gold_labels, modelname)
        classify_data(ml_model, vec, inputfile, outputfile.replace('.conll','.' + modelname + '.conll'))

    
if __name__ == '__main__':
    main()


