# random-graph-nn-paper

Code and data for the paper:

### Romuald A. Janik, Aleksandra Nowak, *Analyzing Neural Networks Based on Random Graphs*, [arXiv:2002.08104](https://arxiv.org/abs/2002.08104)
*We perform a massive evaluation of neural networks with architectures corresponding to random graphs of various types. We investigate various structural and numerical properties of the graphs in relation to neural network test accuracy. We find that none of the classical numerical graph invariants by itself allows to single out the best networks. Consequently, we introduce a new numerical graph characteristic that selects a set of quasi-1-dimensional graphs, which are a majority among the best performing networks. We also find that networks with primarily short-range connections perform better than networks which allow for many long-range connections. Moreover, many resolution reducing pathways are beneficial. We provide a dataset of 1020 graphs and the test accuracies of their corresponding neural networks [at this https URL.](https://github.com/rmldj/random-graph-nn-paper)*
# Repository Structure.

The repository consists of the following folders:

- `src` - the main code.
    - `main.py` - the main program used to run simulations.
    - `analyze` - code for generating network features (see [Generating network features](#generating-network-features)).
    - `elementary_modules.py` - the file containing the definitions of the graph nodes operations (ie. Input, Output,
     Reduce and Node modules).
    - `train` - the directory containing `simulate.py`, which defines the training procedure. 
    - `models` - the directory containing the graph families generators and functions necessary to transform them to artificial neural networks. 
    - `describe` - code for plotting the graphs.
    - `run` - some scripts for quick-start.
    - `data` - The main program will download to this directory (and create it if necessary) CIFAR10 dataset upon first use (unless other data path is specified). This is also the place where the fMRI data may be stored (see [Creating fMRI based graphs](#creating-fmri-based-graphs)).
    
- `graphs` - the directory with an example architecture. See [Downloading the Neural Networks associated with the Graphs](#downloading-the-neural-networks-associated-with-the-graphs) and [Creating the Graphs](#creating-the-graphs) for instructions on how to obtain other architectures.
- `reports` - the directory with datasets. See [Available Datasets](#available-datasets) for summary.


# Running the Code.

To download the repository navigate to a chosen folder and run:

```
git clone https://github.com/rmldj/random-graph-nn-paper.git
```

Please note that all of the below examples
assume that ***the scripts are run from the main repository directory***, which is also ***added to the*** `PYTHONPATH` (otherwise the scripts and imports may need some manual adjustments):
  
```
export PYTHONPATH="<path-to-the-directory-with-the-repository>/random-graph-nn-paper":$PYTHONPATH
```
example:

```
export PYTHONPATH="/home/someusername/projects/random-graph-nn-paper":$PYTHONPATH
```
 


The code is written using python 3.6. In ordered to be able to run the programs, the following packages are required:


```
numpy==1.16.2
matplotlib==3.0.2
scipy==1.2.1
pandas==1.0.1
torch==1.2.0
torchvision==0.2.2
sympy==1.4
networkx==2.3
```

The listed versions of the packages are the versions used by the authors. 
Any higher versions should probably work as well. 
To produce the fMRI graphs one needs to install the `nibabel` package as well. For more information on creating the fMRI graphs see [Creating fMRI-based Graphs](#creating-fmri-based-graphs). 

Navigate to the `random-graph-nn-paper` directory and try to run the example script:
```
cd random-graph-nn-paper
./src/run/example_run.sh er1kx_10_30_v0 
```

This code will train the example architecture defined in `graphs/er1kx_10_30_v0.py` and save the results in `example_run/er1kx_10_30_v0` (if the directory does not exists, it will be created). 
You may replace `er1kx_10_30_v0` with any other random graph architecture (see [Creating the Graphs](#creating-the-graphs)).

Note that you may also want to run the `main.py` script directly, to be able to adjust some command line options. See:
```
python src/main.py --help
```

# Downloading the Neural Networks associated with the Graphs

The PyTorch code for the 1020 neural networks associated to random graphs used in the paper can be downloaded from https://doi.org/10.5281/zenodo.3700845. Put these graphs in the folder `graphs/`. Each network has fields which allow to reconstruct the corresponding `networkx` graph together with a preferred 2D embedding as well as a Python dictionary with the parameters which were used to generate the given graph/neural network. The relevant graph can be easily displayed (see [Plotting a Graph](#plotting-a-graph)). Below, for completeness, we also give instructions for recreating these graphs from scratch.

# Creating the Graphs.

In order to recreate the `er`, `ws`, `ba`, `rdag` and `composite` graphs (for the definitions of those families please refer to the paper)
run: 

```
./src/run/make_graphs.sh
```

This command may take few minutes. 

# Creating the Bottleneck Variants.

To create the bottleneck variant of a graph architecture stored in arbitrary `<filename>.py` 
run:

```
python src/models/bottleneck.py <filename>
```   

Note that this code assumes that `<filename>.py` is stored in the directory `./graphs`. If this is not true, specify the directory with the `--net-dir` option:

```
python src/models/bottleneck.py <filename> --net-dir <directory-where-filename.py-is-stored>
```   

For help see:

```
python src/models/bottleneck.py --help
```


# Creating fMRI-based Graphs.

#### 1. Download the data.
The resting state fMRI connectome data can be downloaded from https://db.humanconnectome.org after registering and accepting data use terms.
Once you login, open the dataset **WU-Minn HCP Data - 1200 Subjects** and download the package **HCP1200 Parcellation+Timeseries+Netmats (PTN)** (choose the version with 1003 subjects, with size circa 13GB). 

Move the file to a chosen location and unzip it:

```
unzip HCP1200_Parcellation_Timeseries_Netmats.zip
```

Enter the `HCP_PTN1200` folder and unzip the `netmats` data with dimension 50 and 100:

```
tar -xzvf netmats_3T_HCP1200_MSMAll_ICAd50_ts2.tar.gz;
tar -xzvf netmats_3T_HCP1200_MSMAll_ICAd100_ts2.tar.gz 
``` 

this will create the `netmats` folder with two subfolders (3T_HCP1200_MSMAll_d100_ts2 and 3T_HCP1200_MSMAll_d50_ts2), which 
contain the data. Move the `netmats` folder to the `data` folder in this project (Alternatively, change the data path in the `src/models/fmri.py` file). 

#### 2. Install `nibabel`.

In order to create the fMRI-based graphs, you will need the `nibabel` package.
This can be done via pip (see https://nipy.org/nibabel/installation.html):

```
pip install nibabel
```

#### 3. Install `Graph_Sampling`.

The obtained by thresholding graphs are subsampled in order to match the standard number of nodes used in the experiments
(see **Section 4.3** in the paper for more info). This is done with the use of algorithms from https://github.com/Ashish7129/Graph_Sampling. 
Follow the instructions provided in link in the section "Installing the development version" in order to install the package. 

#### 4. Run the code.

Now you may run the code for creating the `fmri` class of graphs:
```
python src/models/make_graphs_fmri.py
```

# Plotting a Graph.

You may plot any graph using the `./src/run/describe.sh` script and providing the architecture name. For example:

```
./src/run/describe.sh er1kx_10_30_v0
```

or use the python script directly:

```
python src/describe/describe.py er1kx_10_30_v0
```

This will plot the er1kx_10_30_v0 graph. To save a .png file instead of plotting you maye use:

```
python src/describe/describe.py er1kx_10_30_v0 --save
```

You may change the network directory (containing the specified architecture) or the save directory using command line arguments. See:

```
python src/describe/describe.py --help
``` 



# Available Datasets.

We trained and evaluated 1020 different network architectures on the CIFAR10 dataset. For each of the network we computed 54 network features.
See in the paper:
- ***supplementary material A*** for the training regime.
- ***supplementary material F*** for the summary of trained network models.
- ***suplementary material C*** for the definitions of the computed features and the applied data cleaning procedures.  

#### 1. Computed Graph Features form Network Analysis. 
We store the computed (and cleaned) network features in the `reports` directory. This includes:

- `reports/data30.pkl` - the cleaned dataset of network features computed for graphs with 30 nodes. 
Contains a dictionary with the train and test splits. 
- `reports/data60.pkl` - the cleaned dataset of network features computed for graphs with 60 nodes. 
Contains a dictionary with the train and test splits. 

The train and test splits were used to perform regression (with the CIFAR10 test accuracy as the target variable and network features as predictors - see ***supplementart materials D***). If you do not intend to use the splits, you may concatenate the data.  

To load the datasets using pandas (example for the 30 nodes, `pd` means `pandas`):
```
data = pd.read_pickle('./reports/data30.pkl')
```
now `data` is a dictionary with keys:

- `"X_train"` - the train split.
- `"X_test"` - the test split.
- `"y_train"` - the train target (test accuracy on CIFAR10).
- `"y_test` - the test target (test accuracy on CIFAR10).
   
In addition, we provide data averaged over the versions of the model, available by providing one of the above keys with suffix "`_avg`" (i.e. `data["y_train_avg"]` will return the train targets averaged over the versions of the models). 

If you don't need the train/test splits, the data can be easily concatenated using:

```
df = pd.concat(data["X_train"],data["X_test"])
df_y = pd.concat(data["y_train"],data["y_test"])
```
#### 2. Test accuracies on CIFAR10 and CIFAR100

In addition, the CIFAR10 test accuracies (generalization performance for each network) are stored in the `"./reports/dfresults.pkl"` file.
To load the file with pandas use:

```
df_results = pd.read_pickle('./reports/dfresults.pkl')
```

Now `df_results` is a table with the index being the name of the architecture, and the columns containing:
- `test_acc` - the test accuracy on CIFAR10.
- `test_acc_last10` - the test accuracy on CIFAR10 evaluated and averaged over last 10 epochs. 

We have also evaluated selected models with $60$ nodes on the CIFAR100 datasets. Those results may be accessed by the `./reports/dfresults100.pkl` file.

# Generating Network Features

If you wish to produce the raw (uncleaned) network features, this can be obtained by running the `src/run/analyze.sh` script:
```
./src/run/analyze.sh
```

or directly envoking the underlying python file:
```
python src/analyze/make_df.py
```

This will (by default) compute the network analysis for all graphs listed in the index of `reports/dfresults.pkl`. The results will be stored in `reports/dffeaturestotal.pkl`.
If you wish to change the input or output filepath, this can be done by command line arguments. See:

```
python src/analyze/make_df.py --help
``` 

Computing the raw features may take a while (more than one hour).

Note that this program returns the ***raw*** data! To use the clean dataset refer to [Available Datasets](#available-datasets). For the list of finally used predictors and the applied postprocessing refer to ***suplementary material C*** in the paper. 

To compute the features used to select the 
best architectures in the paper use:

```
python src/analyze/newdata.py
```

This will compute the  `n_bottlenecks` and `pca_elongation` features for all graphs with indexes in `reports/data30.pkl` 
and `reports/data60.pkl`, keeping the train/test split. The results will be saved in 
`reports/newdata30.pkl` and `reports/newdata60.pkl`. To change the output directory use the `--output` command line argument, see:

```
python src/analyze/newdata.py --help
``` 

# Additional Notes

If you have any problems with using the code don't hesitate to fill an issue on github and/or conntact the authors directly.
  
##### 

---

The fMRI partial correlation matrix data were provided by the Human Connectome Project (https://www.humanconnectome.org/), 
WU-Minn Consortium(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) 
funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research;
and by the McDonnell Center for Systems Neuroscience at Washington University. 


For the ResNet implementation on CIFAR-10, we used the code by Yerlan Idelbayev available at https://github.com/akamaster/pytorch_resnet_cifar10.

---


