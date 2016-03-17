python adult_dbn.py -h   --- Just do this to see help with all default parameter settings.
You can always tweak the parameters as beow.
An example way to run this is below:
python adult_dbn.py --num_hidden_units 20 --learning_rate 0.001 --noises '[0.1,0.1,0.0]' --epochs 20
In the above, we have a NN with 20 hidden unis, Learning rate 0.001 and noise std deviations 0.1, 0.1, 0.0 corresponding to input, hidden layer weights respectively and is trained for 20 epochs.

If you look at adult_dbn.py, you will find code to load the dataset, stratify split it into 5 different sets. The code will print out the meausure you asked to be printed out.
