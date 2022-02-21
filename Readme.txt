## Deliverables:

Task 1 scripts are under directory model (nb.py, dt.py, rf.py, svc.py)
Task 2 scripts is a1.py
Task 3 report is A1_report

## Usage

Cd into this folder. Enter the following commands into terminal.

1. To see the list of available models 
	python3 a1.py --list-models
	// return: decision_tree naive_bayes random_forest svc_linear svc_sigmoid svc_poly svc_rbf


2. To see the list of available code smells 
	python3 a1.py --list-smells
	// return: long_method god_class feature_envy data_class


3. To compare the accuracy and F1 score received on test set and training set of a code smell using a model
	python3 a1.py --compare <MODEL> <SMELL>
	ex: python3 a1.py --compare decision_tree long_method


4.  To see the accuracy and F1 score on test set of 1 or more code smells using a model
	python3 a1.py --run <MODEL> <SMELL> [<SMELL>...]
	ex: python3 a1.py --run decision_tree god_class feature_envy
