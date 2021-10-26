# Graph Neural Net Example

A simple example of node classification using a simple toy model of a displaced vertex.

The network performs two tasks: classifying tracks into signal (red), and background (blue), and predicts the position of the signal vertex.

The implementation is far from perfect, clearest example is the handeling of the two sets of labels. The network is not at all optimised, this is just the first architecture that kinda worked.

## Example output

![plot](example/example.png)


## Setting up

Note, could be gaps in this setup procedure. 

### Step 1: obtaining conda on DICE

* ssh to DICE IPU login node
```
ssh USER@gc00.dice.priv
```
* Run the following to set up conda - a package manager - on the DICE node
```
wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
CONDA_INSTALL_PATH=/software/$USER/miniconda # or /storage/$USER/miniconda
bash miniconda.sh -b -p ${CONDA_INSTALL_PATH}
PATH=${CONDA_INSTALL_PATH}/bin:$PATH; export PATH
rm -f miniconda.sh
conda update conda -y
conda update pip -y
```
* Add the following to your bashrc file (file that gets called everytime you login to DICE). You can use nano to edit the file. (crtl-x, y, enter) to close and save.
```
nano ~/.bashrc
export PATH=/software/$USER/miniconda/bin:$PATH
* Then source .bashrc with
```
source ~/.bashrc
```
* Check conda is installed, the following should return a file path. For example "/software/USER/miniconda/bin/conda"
```
which conda
```
* Then create a conda environment with the following
```
conda create --prefix /users/USER/conda python=3.6
```

### Step 2: setting up Graphcore SDK

* Obtain copy of Graphcores Poplar Software Development Kit (SDK), for example "poplar_sdk-centos_7_6-2.2.0-7a4ab80373.tar.gz"
* Create a directory to work in
```
mkdir IPU
```
* Copy SDK file to this area
```
scp poplar_sdk-centos_7_6-2.2.0-7a4ab80373.tar.gz USER@gc00.dice.priv:/users/USER/IPU/.
```
* Unzip and install 
```
tar -xvzf poplar_sdk-centos_7_6-2.2.0-7a4ab80373.tar.gz
cd poplar_sdk-centos_7_6-2.2.0+688-7a4ab80373
source poplar-centos_7_6-2.2.0+166880-feb7f3f2bb/enable.sh
source popart-centos_7_6-2.2.0+166880-feb7f3f2bb/enable.sh
```
* Enter the conda environment we created earlier with
```
source activate /users/USER/conda
```
* Create and activate a Python virtual environment, a requirement to use Graphcore's Tensorflow, with
```
virtualenv -p python3.6 ~/workspace/tensorflow_env
source ~/workspace/tensorflow_env/bin/activate
```
* Install Graphcore's tensorflow with
```
pip install tensorflow-2.4.1+gc2.2.0+79539+c6a61d7b19a+intel_skylake512-cp36-cp36m-linux_x86_64.whl
```
* Check all is well with 
```
python -c "from tensorflow.python import ipu"
```
* Install spektral with 
```
pip install --upgrade pip
pip install spektral
```

### Step 3: Create source.sh 

* In your home directory ('cd') create a file called source.sh (nano source.sh) with the following inside
```
source ~/IPU/poplar_sdk-centos_7_6-2.2.0+688-7a4ab80373/poplar-centos_7_6-2.2.0+166880-feb7f3f2bb/enable.sh
source ~/IPU/poplar_sdk-centos_7_6-2.2.0+688-7a4ab80373/popart-centos_7_6-2.2.0+166880-feb7f3f2bb/enable.sh
source ~/workspace/tensorflow_env/bin/activate
```
* Everytime you log on you need to run the following quick command to set up libraires (or can add it to your .bashrc)
```
source source.sh
```

### Step 4: cloning this repository

* In you IPU working directory clone this repository
```
cd ~/IPU/
git clone https://github.com/alexmarshallbristol/GNN_example.git
```
* Run the code
```
python GNN.py
```

Cheers :smiley:














