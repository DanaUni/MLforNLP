# This file is a combination of adjusted functions from the basic_evaluation.ipynb file from the course ML for NLP.

import argparse
import pandas as pd
from collections import defaultdict, Counter
import ner_machine_learning

system_column_identifier = 8

def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    """
    This function extracts annotations represented in the conll format from a file

    :param inputfile: the path to the conll file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :type inputfile: string
    :type annotationcolumn: string
    :type delimiter: string

    :returns: the annotations as a list
    """
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    conll_input = pd.read_csv(inputfile, sep=delimiter, quotechar=delimiter, header=None, on_bad_lines='skip')
    annotations = conll_input[int(annotationcolumn)].tolist()

    return annotations


def obtain_counts(goldannotations, machineannotations):
    """
    This function compares the gold annotations to machine output

    :param goldannotations: the gold annotations
    :param machineannotations: the output annotations of the system in question
    :type goldannotations: the type of the object created in extract_annotations
    :type machineannotations: the type of the object created in extract_annotations

    :returns: a countainer providing the counts for each predicted and gold class pair
    """
    # TIP on how to get the counts for each class
    # https://stackoverflow.com/questions/49393683/how-to-count-items-in-a-nested-dictionary, last accessed 22.10.2020
    evaluation_counts = defaultdict(Counter)
    for i in range(len(goldannotations)):
        evaluation_counts[goldannotations[i]][machineannotations[i]] += 1

    return evaluation_counts


def provide_confusion_matrix(evaluation_counts):
    """
    Read in the evaluation counts and provide a confusion matrix for each class

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts

    :returns: a confusion matrix
    """
    confusion_dict = dict()
    for gold_key in evaluation_counts.keys():
        confusion_dict[gold_key] = evaluation_counts[gold_key]

    confusion_matrix = pd.DataFrame.from_dict(confusion_dict, orient='index').fillna(0)

    # Drop the O class if required
    if exclude_O_class:
        confusion_matrix = confusion_matrix.drop(columns=['O'])
        confusion_matrix = confusion_matrix.drop('O')

    return confusion_matrix


def tp_tn_fp_fn(confusion_matrix, class_name):
    """
    Calculate the number of true positives (tp), true negatives (tn), false positives (fp) and false negatives (fn) from a confusion matrix and return them in a dictionary

    :param confusion_matrix: the confusion matrix
    :param class_name: the name of the class for which the tp, tn, fp and fn are requested
    :type confusion_matrix: a pandas DataFrame (as returned by provide_confusion_matrix)
    :type class_name: string

    :returns: a dictionary of tp, tn, fp and fn
    """
    tp = confusion_matrix.loc[class_name, class_name]
    fp = sum(confusion_matrix.loc[:, class_name].to_list()) - tp
    fn = sum(confusion_matrix.loc[class_name, :].to_list()) - tp
    tn = confusion_matrix.values.sum() - tp - fp - fn

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def calculate_precision_recall_fscore(evaluation_counts):
    """
    Calculate precision recall and fscore for each class and return them in a dictionary

    :param evaluation_counts: a container from which you can obtain the true positives, false positives and false negatives for each class
    :type evaluation_counts: type of object returned by obtain_counts

    :returns: the precision, recall and f-score of each class in a container
    """

    precision_recall_fscore_dict = dict()
    confusion_matrix = provide_confusion_matrix(evaluation_counts)

    for class_label in confusion_matrix.columns:
        precision_recall_fscore_dict[class_label] = {}

        tp_tn_fp_fn_dict = tp_tn_fp_fn(confusion_matrix, class_label)
        tp = tp_tn_fp_fn_dict['TP']
        tn = tp_tn_fp_fn_dict['TN']
        fp = tp_tn_fp_fn_dict['FP']
        fn = tp_tn_fp_fn_dict['FN']

        precision = tp / (fp + tp)
        recall = tp / (fn + tp)

        # Check if there are any True Postives
        if tp == 0:
            print(f"Warning: Number of True Positives is 0 for class {class_label}")
            fscore = 0
        else:
            fscore = (2 * precision * recall) / (precision + recall)

        # Calculate metrics
        precision_recall_fscore_dict[class_label]['precision'] = precision
        precision_recall_fscore_dict[class_label]['recall'] = recall
        precision_recall_fscore_dict[class_label]['f-score'] = fscore

    return precision_recall_fscore_dict


def carry_out_evaluation(gold_annotations, systemfile, systemcolumn, delimiter='\t'):
    """
    Carries out the evaluation process (from input file to calculating relevant scores)

    :param gold_annotations: list of gold annotations
    :param systemfile: path to file with system output
    :param systemcolumn: indication of column with relevant information
    :param delimiter: specification of formatting of file (default delimiter set to '\t')

    returns: evaluation information for this specific system
    """
    system_annotations = extract_annotations(systemfile, systemcolumn, delimiter)
    evaluation_counts = obtain_counts(gold_annotations, system_annotations)
    confusion_matrix = provide_confusion_matrix(evaluation_counts)

    # Check if the table is to be printed in latex format
    if print_latex:
        print("\nConfusion matrix:\n(gold rows, machine columns)\n", confusion_matrix.to_latex())
    else:
        print("\nConfusion matrix:\n(gold rows, machine columns)\n", confusion_matrix)

    evaluation_outcome = calculate_precision_recall_fscore(evaluation_counts)

    return evaluation_outcome

def calculate_macro_averages(evaluation_outcome, system_and_model_name):
    """
    Calculate the macro averages (mean) of the precision, recall and f-scores over the classes.

    :param evaluations_outcome: the outcome of evaluating a system
    :param system_and_model_name: string, combination of the name of the current system and the model

    :returns: a dictionary with the macro averages of precision, recall and f-score
    """
    # initialise variables
    sum_precision = 0
    sum_recall = 0
    sum_fscore = 0
    count_classes = 0
    macro_metrics_dict = dict()

    # Sum metrics
    for class_name in evaluation_outcome:
        count_classes += 1
        sum_precision += evaluation_outcome[class_name]['precision']
        sum_recall += evaluation_outcome[class_name]['recall']
        sum_fscore += evaluation_outcome[class_name]['f-score']

    # Calculate macro average of metrics (divide sum by count)
    macro_metrics_dict['precision'] = sum_precision / count_classes
    macro_metrics_dict['recall'] = sum_recall / count_classes
    macro_metrics_dict['f-score'] = sum_fscore / count_classes

    # Create macro average dataframe
    macro_avg_df = pd.DataFrame.from_dict({system_and_model_name: macro_metrics_dict}, orient='index')

    # Check if the the table is to be printed in latex format
    if print_latex:
        print("\nMacro averages\n", macro_avg_df.to_latex(float_format="%.3f"))
    else:
        print("\nMacro averages\n", macro_avg_df)

    return macro_metrics_dict

def provide_output_tables(evaluations):
    """
    Create tables based on the evaluation of various systems

    :param evaluations: the outcome of evaluating one or more systems
    """
    # https:stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    evaluations_pddf = pd.DataFrame.from_dict({(i, j): evaluations[i][j]
                                               for i in evaluations.keys()
                                               for j in evaluations[i].keys()},
                                              orient='index')

    # Check if the table is to be printed in latex format
    if print_latex:
        print("\nevaluations_pddf latex\n", evaluations_pddf.to_latex(float_format="%.3f"))
    else:
        print("\nevaluations_pddf\n", evaluations_pddf)


def run_evaluations(goldfile, goldcolumn, systems):
    """
    Carry out standard evaluation for one or more system outputs

    :param goldfile: path to file with goldstandard
    :param goldcolumn: indicator of column in gold file where gold labels can be found
    :param systems: required information to find and process system output
    :type goldfile: string
    :type goldcolumn: integer
    :type systems: list (providing file name, information on tab with system output and system name for each element)

    :returns the evaluations for all systems
    """
    evaluations = {}
    # not specifying delimiters here, since it corresponds to the default ('\t')
    gold_annotations = extract_annotations(goldfile, goldcolumn)
    for system in systems:
        sys_evaluation = carry_out_evaluation(gold_annotations, system[0], system[1])
        evaluations[system[2]] = sys_evaluation
        calculate_macro_averages(sys_evaluation, system[2])

    return evaluations


def identify_evaluation_value(system, class_label, value_name, evaluations):
    """
    Return the outcome of a specific value of the evaluation

    :param system: the name of the system
    :param class_label: the name of the class for which the value should be returned
    :param value_name: the name of the score that is returned
    :param evaluations: the overview of evaluations

    :returns the requested value
    """
    return evaluations[system][class_label][value_name]


def create_system_information(system_information):
    """
    Takes system information in the form that it is passed on through sys.argv or via a settingsfile
    and returns a list of elements specifying all the needed information on each system output file to carry out the evaluation.

    :param system_information is the input as from a commandline or an input file
    """
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    systems_list = [system_information[i:i + 3] for i in range(0, len(system_information), 3)]
    return systems_list


def main():
    """
    Executes feature ablation for user specified models and features.
    """
    global print_latex
    global system_name
    global exclude_O_class

    # Parser arguments
    parser = argparse.ArgumentParser(description='extract features')
    parser.add_argument('trainingfile', type=str)
    parser.add_argument('inputfile', type=str)
    parser.add_argument('outputfile', type=str)
    parser.add_argument('gold_column_identifier', type=str)
    parser.add_argument('-mod', '--models', type=str, nargs="*", default=['logreg'])
    parser.add_argument('-feat', '--features', type=str, nargs="*", default=["token"])
    parser.add_argument('-pl', '--print_latex', type=str, default="False")
    parser.add_argument('-use', '--use_embeddings', type=str, default="False")
    parser.add_argument('-lmp', '--language_model_path', type=str, default="..\models\GoogleNews-vectors-negative300.bin.gz")
    parser.add_argument('-exo', '--exclude_O_class', type=str, default="False")
    args = parser.parse_args()

    # Argument variables
    trainingfile = args.trainingfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    gold_column_identifier = args.gold_column_identifier
    models = args.models
    features = args.features
    language_model_path = args.language_model_path

    # Join lists as strings of variables that need to be passed to ner_machine_learning.py via argparse arguments.
    system_name = "-".join(features)
    selected_models = ",".join(models)
    selected_features = ",".join(features)

    # Return strings to there boolean format
    use_embeddings = ner_machine_learning.string_to_bool(args.use_embeddings)
    print_latex = ner_machine_learning.string_to_bool(args.print_latex)
    exclude_O_class = ner_machine_learning.string_to_bool(args.exclude_O_class)

    # If embeddings are used, only the token feature wil be used (no mixed representations)
    if use_embeddings:
        selected_features = "token"

    # Train models based on selected models and features
    ner_machine_learning.main([trainingfile, inputfile, outputfile, gold_column_identifier, "-mod", selected_models, "-sfeat", selected_features, "-use", str(use_embeddings), "-lmp", language_model_path])

    # Evaluate the output of the selected models and features
    for model in models:
        print(f"\n\nResults for sytem {system_name} with {model} model:")
        system_info = create_system_information([outputfile.replace(".conll", f".{model}.conll"), system_column_identifier, f"{system_name}-{model}"])
        evaluations = run_evaluations(outputfile.replace('.conll', f".{model}.conll"), gold_column_identifier, system_info)
        provide_output_tables(evaluations)


if __name__ == '__main__':
    main()
