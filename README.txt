Descriptions of directories

images: a folder containing a number of train-test images generated for fitting the KMC data
old_scripts: contains the older versions of a few scripts that exist in the main directory
trained_models: contains hdf5 and pb models that have been trained on the KMC data for the Lotka-Voltera model for each 3-site term
training_sk: trained models intermediate checkpoint files for the OAB model I sent you initially
training_allB3: checkpoint files for neural network models stored in trained_models

Descriptions of python scripts
kmc_LVgenKrun.py: generates the csv used for training these models from the raw kmc data. k_run_prep.csv is intended to be the primary dataframe to train models on. k_run.csv was used while I was still figuring out the models and deciding which features to add permanently
kmc_ml_util.py: utility functions used by the other scripts
kmc_NN_prep.py: neural network script designed to run on the k_run_prep.csv made by kmc_LVgenKrun.py. Uses scikit-learn
kmc_NN.py: an earlier version of kmc_NN_prep.py that runs on k_run.csv. Uses scikit-learn
kmc_tf.py: neural network training script for testing tensorflow for generating 3-site terms
kmc_tf_all3siteLV.py: tensorflow neural network training script to generate models for all 3-site terms, and save them for conversion to C++
kmc_save_graph.py: converts a tensorflow-generated hdf5 file into a .pb file that should be accessible to C++

