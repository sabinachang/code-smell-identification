import argparse
import os.path
from model import model_runner

# map command line arguments into file prefix and suffix
valid_model = {
    'decision_tree': 'dt_',
    'random_forest': 'rf_',
    'naive_bayes': 'nb_',
    'svc_linear': 'svcl_',
    'svc_poly': 'svcp_',
    'svc_rbf': 'svcr_',
    'svc_sigmoid': 'svcs_'
}

valid_smell = {
    'god_class': 'god',
    'data_class': 'data',
    'feature_envy': 'feature',
    'long_method': 'long'
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-models', nargs='?', const=1, dest='list_models',
                        help='list all the trained models')
    parser.add_argument('--list-smells', nargs='?', const=1, dest='list_smells',
                        help='list all the code smells')
    parser.add_argument('--compare', dest='compare',nargs=2,
                        help='--compare <MODEL> <SMELL> : '\
                        'compare the accuracy and F1 score received on testing '\
                        'and training set of SMELL using MODEL')
    parser.add_argument('--run', dest='run',nargs='+',
                        help='--run <MODEL> <SMELL1> [<SMELL2>...] : '\
                        'see the accuracy and F1 score on test set that MODEL achieved for SMELL')

    args = parser.parse_args()
    if args.list_models:
        list_models()
    elif args.list_smells:
        list_smells()
    elif args.compare:
        compare_performance(args.compare)
    elif args.run:
        run_model(args.run)
    else:
        pass

    return

def compare_performance(args):
    # check the validity of arguments
    if (args[0] not in valid_model.keys()):
        print('Model {} is not supported'.format(args[0]))
        return
    
    if (args[1] not in valid_smell.keys()):
        print('Smell {} is not supported'.format(args[1]))
        return

    name = valid_model[args[0]] + valid_smell[args[1]]
    file = 'output/' + name + '.txt'
    if not os.path.isfile(file):
        #train model if no file exists yet 
        model_runner.runners[name]()

    # print metrics from file
    print_compare_info(args[1], file)
    return

def run_model(args):
    # check the validity of arguments
    if (args[0] not in valid_model.keys()):
        print('Model {} is not supported'.format(args[0]))
        return
    if (len(args) == 1):
        print('No smell is given. Please specify one')
        return

    unique = set()
    for arg in args[1:]:
        if (arg not in valid_smell.keys()):
            print('Smell {} is not supported'.format(arg))
            return
        # only put unique smell into final set
        if (arg not in unique):
            unique.add(arg)
    
    # convert model + smell into file names
    files = []
    for u in unique:
        label =  valid_model[args[0]] + valid_smell[u]
        file = 'output/' + label + '.txt'
        if not os.path.isfile(file):
            # train model if no file exists yet
            model_runner.runners[label]()

        files.append((file, u))

    # print metrics from files
    print_run_info(files)

def list_smells():
    print('Available smells include:')
    for k in valid_smell.keys():
        print(k)
    
def list_models():
    print('Available models include:')
    for k in valid_model.keys():
        print(k)

def print_compare_info(smell, file):
    # print header
    label = smell.replace('_', ' ').title()
    print('\n{}      |   Accuracy |   F1-score  '.format(label))

    # print performance metrics
    col = ['Training set      ', 'Test set          ']
    with open(file,'r') as filehandle:
        for i, line in enumerate(filehandle):
            # convert string to percentage
            fields = line.split(' ')
            accuracy = '{:.2%}'.format(float(fields[0]))
            f1 = '{:.2%}'.format(float(fields[1]))
            print('{}|   {}   |   {}   '.format(col[i], accuracy, f1))

    return

def print_run_info(files):
    # print header
    print('\nSmell      |   Accuracy |   F1-score  ')
    
    for file in files:
        # generate smell label 
        label = file[1].replace('_', ' ').title()
        with open(file[0],'r') as filehandle:
            # use second line to get test set performance metrics
            l = filehandle.readlines()
            fields = l[1].split(' ')
            accuracy = '{:.2%}'.format(float(fields[0]))
            f1 = '{:.2%}'.format(float(fields[1]))
            print('{}|   {}   |   {}   '.format(label, accuracy, f1))

if __name__ == "__main__":
    parse_arguments()