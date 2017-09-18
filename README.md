atai
----

Code for [Analyze This! does AI](https://www.meetup.com/AnalyzeThis/) competition

Runs a deep neural network against the notMNIST dataset.

Getting Started
---------------

NOTE: These instructions only cover setting up Keras with CPU `tensorflow`
support. If you want to run this with GPU support, you will want to install the
`tensorflow-gpu` package instead and follow these instructions to install all of
the necessary libraries:

https://www.tensorflow.org/tutorials/using_gpu


Otherwise, you will first need to download the notMNIST data set:

    wget http://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz
    wget http://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz

Next, create a virtual environment and install the Python dependencies:

    # If you have pipenv installed
    pipenv install --three
    pipenv shell

    # If you don't have pipenv installed
    virtualenv venv -p python3
    source venv/bin/activate
    pip install -r requirements.txt

Then, run the preprocessing step to convert the notMNIST datasets into `numpy`
arrays. This will create the files `notmnist_large.npz` and `notmnist_small.npz`.

    python preprocess.py

Finally, run `train.py` to train the model. If not not passed any arguments, it
will default to running against the MNIST data set. To train against one of the
notMNIST data sets, pass in the path to the preprocessed data file.

    python train.py                     # Runs against MNIST data set
    python train.py notmnist_large.npz  # Runs against large notMNIST data set
    python train.py notmnist_small.npz  # Runs against small notMNIST data set

This will show how well the model did against the data set, and then save it as
`model.h5`. To then get predictions from the model later, use `predict.py`
