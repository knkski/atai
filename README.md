atai
----

Code for [Analyze This! does AI](https://www.meetup.com/AnalyzeThis/) competition

Runs a deep neural network against the notMNIST dataset.


Getting Started
---------------

NOTE: These instructions only cover setting up Keras with the slower CPU
support. Installing GPU support is beyond the scope of this README, but
instructions can be found here:

https://www.tensorflow.org/tutorials/using_gpu

ANOTHER NOTE: These instructions reference `~/atai`, which works on Mac. Windows
users should substitute `%userprofile%/data`.

To start, ensure that you have Docker installed:

https://www.docker.com/docker-mac
https://www.docker.com/docker-windows

You will need to restart your computer after installing Docker. Then, install
Kitematic:

https://kitematic.com/

After this, Docker should be up and running. Note that if you ever get an error
such as this, you should start Kitematic to ensure that Docker is running properly:

    Cannot connect to the Docker daemon. Is the docker daemon running on this host?

Next, download this repository by running this command from the terminal:

    git clone https://github.com/knkski/atai.git
    cd atai

You will then need to download the notMNIST datasets. You can either run these
commands, or copy the URLs into your browser and move the downloaded files to
the directory that the code lives in

    wget http://commondatastorage.googleapis.com/books1000/notMNIST_large.tar.gz
    wget http://commondatastorage.googleapis.com/books1000/notMNIST_small.tar.gz


Preprocessing
-------------

You will need to extract the dataset files into normal image files. On Windows,
you can install [7-zip](http://www.7-zip.org/download.html) to extract them. On
Mac, you can run this command:

    tar -xzvf ./notMNIST_small.tar.gz
    tar -xzvf ./notMNIST_large.tar.gz

Then, run the preprocessing step to convert the notMNIST datasets into `numpy`
arrays. This will create the files `notmnist_large.npz` and `notmnist_small.npz`.

    docker run -v ~/atai:/atai --rm knkski/atai python preprocess.py

Note that you should (hopefully) only have to do this once, unless you start
playing around with modifying the datasets yourself.


Training
--------

Finally, run `train.py` to train the model. If not not passed any arguments, it
will default to running against the MNIST data set. To train against one of the
notMNIST data sets, pass in the path to the preprocessed data file.

    docker run -v ~/atai:/atai --rm knkski/atai python train.py                     # Runs against MNIST data set
    docker run -v ~/atai:/atai --rm knkski/atai python train.py notmnist_large.npz  # Runs against large notMNIST data set
    docker run -v ~/atai:/atai --rm knkski/atai python train.py notmnist_small.npz  # Runs against small notMNIST data set

This will show how well the model did against the data set, and then save it as
`model.h5`.


Predicting
----------

After generating a `model.h5` file, you can use that file to make predictions of
whatever image files you wish with this command:

    docker run -v ~/atai:/atai --rm knkski/atai python predict.py notMNIST_small/A/MDEtMDEtMDAudHRm.png


Jupyter
-------

You can run a Jupyter notebook with this codebase if you wish. Run this command,
and copy/paste the link that it gives you into your browser:

    docker run -v ~/atai:/atai --rm -p 8888:8888 knkski/atai jupyter notebook --allow-root

To stop Jupyter, simply press Ctrl+C in the terminal window where you ran the
command.


Tensorboard
-----------

Tensorboard is a built-in tool for Tensorflow, and has a bunch of cool
visualizations available. You can it with this command:

    docker run -v ~/atai:/atai --rm -p 6006:6006 knkski/atai tensorboard --logdir=logs/

To stop tensorboard, you can also press Ctrl+C in the terminal window you ran it
from.


Background Reading
------------------

These papers have been recommended for background on deep learning:

https://arxiv.org/pdf/1611.00847.pdf

https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf

https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
