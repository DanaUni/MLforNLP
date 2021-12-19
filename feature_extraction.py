# This file is a combination of adjusted functions from the prerpocessing_conll_2021.ipynb file for the course ML for NLP.

import argparse

def read_in_conll_file(conll_file: str, delimiter: str = '\t'):
    """
    Reads in conll file and returns structured object

    :param conll_file: path to conll_file
    :param delimiter: specifies how columns are separated. Tabs are standard in conll

    :returns: List of splitted rows included in conll file
    """
    conll_rows = []
    with open(conll_file, 'r') as my_conll:
        for line in my_conll:
            row = line.strip("\n").split(delimiter)
            if len(row) == 1:
                conll_rows.append([""]*rowlen)
            else:
                rowlen = len(row)
                conll_rows.append(row)

    return conll_rows

def read_in_toponomy(toponomyfile):
    """
    Reads in the toponomy suffixes file and returns a list of the suffixes in the file

    :param toponomyfile: input file with the toponomy suffixes

    :returns: a list of suffixes
    """
    global average_suffix_length

    suffixes = []
    with open(toponomyfile, 'r') as my_toponomy:
        next(my_toponomy)
        for line in my_toponomy:
            suffixes.append(line.strip())

    # Calulate the average (mean) length of the place name suffixes.
    max_len_suffix = len(max(suffixes, key=len))
    min_len_suffix = len(min(suffixes, key=len))
    average_suffix_length = (max_len_suffix + min_len_suffix)/2

    return suffixes

def extract_features(conll_rows: list, toponomy_suffixes: list):
    """
    Extracts features from the tokens in the conll file

    :param conll_rows: rows of the conll file
    :param toponomy_suffixes: a list of common place name suffixes

    :returns: a list of rows with added features, ready to turn into a conll
    """
    conll_rows_out = []
    for row in conll_rows:
        token = row[0]
        if len(token) > 0:

            # Feature for whether or not the token is contains capital letters.
            if token.istitle():
                row.append("FirstCap")
            elif token.isupper():
                row.append("AllCaps")
            elif token.islower():
                row.append("NoCaps")
            else:
                row.append("Rest")

            # Feature for whether or not the token contains a hyphen.
            if "-" in token:
                row.append("Hyphen")
            else:
                row.append("NoHyphen")

            # Feature for whether or not the token consists of digits.
            if token.isdigit():
                row.append("AllDigits")
            else:
                row.append("NotAllDigits")

            # Feature to check whether or not the suffix of a token is common place name suffix (advanced feature)
            if len(token) >= average_suffix_length:
                token_suffix = token[-4:].lower()

                for toponomy_suffix in toponomy_suffixes:
                    if (token_suffix == toponomy_suffix) or (token_suffix in toponomy_suffix) or (toponomy_suffix in token_suffix):
                        row.append("ToponomySuffix")
                        break
                else:
                    row.append("NoToponomySuffix")
            else:
                row.append("NoToponomySuffix")

        else:
            row = ""

        conll_rows_out.append(row)

    return conll_rows_out


def write_outputfile_with_features(conll_rows_out: list, outputfilename: str, delimiter: str = '\t'):
    """
    Writes a conll file from conll rows with features.

    :param conll_rows_out: a list of rows with features
    :param outputfilename: path of outputfile
    :param delimiter: delimiter to separate the features
    """
    with open(outputfilename, 'w') as outputfile:
        for row in conll_rows_out:
            outputfile.write(delimiter.join(row) + "\n")

def main():
    """
    Runs the feature extraction on an inputfile and outputs the file with added features.
    """

    # Parser arguments
    parser = argparse.ArgumentParser(description='extract features')
    parser.add_argument('inputfile', type=str)
    parser.add_argument('outputfile', type=str)
    parser.add_argument('-top', '--toponomyfile', type=str, default='settings\generic_suffixes_toponomy_UK_IR.txt')
    args = parser.parse_args()

    # Argument variables
    inputfile = args.inputfile
    outputfile = args.outputfile
    toponomyfile = args.toponomyfile

    # Read in the inputfile and toponomy suffixes file
    no_features_inputfile = read_in_conll_file(inputfile)
    toponomy_suffixes = read_in_toponomy(toponomyfile)

    # Extract the features from the tokens in the inputfile
    conll_rows_with_features = extract_features(no_features_inputfile, toponomy_suffixes)

    # Write a conll outputfile that contains the conll rows with the added features
    write_outputfile_with_features(conll_rows_with_features, outputfile)


if __name__ == '__main__':
    main()