# SAWO-NN repository

Welcome to the SAWO-NN repository on Github. This code is considered supplementary material for a journal article published in the Springer journal of Neural Computing and Applications in March 2022, available at https://link.springer.com/article/10.1007/s00521-022-07035-5. This README describes the code artifacts of the project, how to install the project onto your Google Drive, and how to run the project using Google Colaboratory.

## Code description
- **control** folder: This folder contains JSON files, used to enable the parallelizability of grammatical evolution. The Controller notebook writes to these files before each generation. The Runner notebooks read this information and evaluate population members, storing their results in these files. The Controller notebook then reads the evaluations, and uses them to create the next generation, and so on.
- **datasets** folder: This folder contains pre-processed input and target NumPy arrays for each of the 12 datasets. A note here that the dataset IDs are slightly different as to in the paper, as described in the table below. Therefore, to reproduce results for a particular dataset ID from the paper, please use the corresponding code ID.

| Paper ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Code ID | 1 | 2 | 3 | 5 | 6 | 7 | 8 | 10 | 11 | 12 | 4 | 9 |

- **runs** folder: This folder will contain JSON files generated from every full run of the grammatical evolution, created by the Controller notebook.
- **samples** folder: This folder will contain JSON files for every generated sample, created by the SampleComparison notebook.
- **Architecture.py**: Instances of this Python class are instantiated using a shorthand-architecture-array, which dynamically builds the corresponding Tensorflow architecture, as per the architecture construction rules of the paper. This class also provides a method to calculate the error currently produced by the built architecture on some inputs and targets.
- **Controller.ipynb**: This Jupyter notebook controls the flow of the grammatical evolution, as well as gives instructions to the Runner notebooks. It generates the initial population, and repeatedly does the following: assigns the population amongst the Runner notebooks, waits for them to evaluate their assigned members, captures the evaluations, and performs selection and application of genetic operators to create a new population. While doing all of this, it captures summary information about fitness values, genetic diversity and genetic operator effectiveness. Finally, it saves all this information and the best found members over the run into a JSON file, stored in the **runs** folder.
- **Member.py**: This Python class calls the necessary functions of **RuleSet.py** to convert a bitstring chromosome into the information needed to instantiate an instance of **SawoNN.py** through the grammar, and instantiates it. It then performs the repeated training of this SAWO-NN on provided inputs and targets, calculating and returning fitness based on the average generalization abilities after these trainings.
- **Node.py**: This Python class is used in determining which sets-of-weights need to be maintained for a SAWO-NN based on its chromosome. It keeps reference of edges from it to other nodes, and can find all nodes reachable from it by a directed walk.
- **Rule.py**: This Python class is used to logically store all possible outputs from one symbol of the grammar, and is instantiated based on the grammar provided.
- **RuleSet.py**: This Python class takes a text file containing grammar rules as input, parses the file, and generates a full set of grammar rules, each an instance of **Rule.py**. Using this full set of rules, it can take input bitstring codons, use these bitstring codons to choose rule options as per the technique of grammatical evolution, and return the generated instance of the grammar correlating to that input. 
- **Runner_1...4.ipynb**: These Jupyter notebooks perform evaluation of population members assigned to them by the Controller notebook. This is done by instantantiating an instance of **Member.py** for every bitstring chromosome it has been assigned, calling the fitness function of this class, and storing these fitnesses into the control files for the Controller notebook to read. There are 4 notebooks because Google Colab allows for a maximum of 5 workbooks to be running concurrently. These notebooks are identical, except for one variable *me*, which is uniquely set for each notebook and allows the notebook to find its assigned members.
- **SampleComparison.ipynb**: This Jupyter notebook is used to generate the performance sample for a particular bitstring chromosome (and the SAWO-NN it describes) on a particular dataset. The best chromosomes for each dataset are provided, so their performance samples can be re-generated and saved in the **samples** folder in JSON format, however, their pre-computed performance samples are also provided, to save time. Performance samples for the baseline models can also be generated, with the best learning rates found for each dataset also provided. Finally, the performance sample of the best SAWO-NN can be compared to the baseline model performance sample, for each dataset, to re-confirm the hypothesis testing results achieved.
- **SawoNN.py**: This Python class implements a SAWO-NN as described in the paper. To instantiate a SAWO-NN, information must be provided on layer sizes, the primary architecture, and each weight update equation. The needed weights and architectures can then be calculated. To train a SAWO-NN on some inputs and targets, the training process in the paper is followed. This class implements the important function used to calculate the error derivative of any set-of-weights in any initialised architecture, thanks to some clever usage of Tensorflow's GradientTape and the **Architecture.py** class.
- **grammar.txt**: This text file contains the grammar rules specified in the paper, albeit with symbols linked to a single output replaced with that output. This avoids redundant codons without decreasing the possible search space. This file is read and parsed by the **RuleSet.py** class.

## Installation on Google Drive
The only pre-requisites for installation is that you have a Google account, with Google Drive. The only web browser we have used for this is Google Chrome, but we think any web browser should work - but we cannot confirm this. Please follow the below steps to install the project:
1. From the homepage of this repository, click the green Code button and select the 'Download Zip' dropdown, which will download a zipped version of the project to your computer. 
2. Unzip the zipped project, which will produce a sawo-nn-main folder. Go into the folder. You will see another folder called sawo-nn-main. Go into this folder. You will then see a multitude of folders and files, as described in the previous section. This is the source code.
3. Go to the homepage of your Google Drive using Google Chrome i.e. the 'My Drive' folder. If a folder named 'Colab Notebooks' does not already exist, create it, and go into this folder. Create a new folder called 'TestSAWO'. Go into this folder.
4. Upload all the files in the folder reached in step 2 (Your PC/sawo-nn-main/sawo-nn-main/\*) to the current folder you are in on your Google Drive (My Drive/Colab Notebooks/TestSAWO). You can do this by dragging and dropping all the files from your file explorer, or by manually uploading using the 'New' button on Google Drive.
5. Once the upload is complete, the project is installed on your Google Drive.

## Running the project with Google Colaboratory
Nothing needs to be manually installed to run the project, as Google Colaboratory (or 'Colab') handles all necessary package installation and Tensorflow configuration. There are two ways to run this project: perform a run of the grammatical evolution to search for optimal SAWO-NNs, or create a performance sample of a found SAWO-NN to compare against the optimized baseline model for a particular dataset.

### Grammatical evolution search
1. In the My Drive/Colab Notebooks/TestSAWO folder on Google Drive, double click on the following notebooks to open them: Controller, Runner_1, Runner_2, Runner_3, Runner_4. Wait for them all to load.
2. Navigate to Controller. This is used to launch a run. You can edit some hyperparameters near the top of the notebook, as specified by the comments in the code.
3. Click the 'Runtime' dropdown, and select 'Run all'. The first code block involves getting access to your Google Drive, and on your first time doing this, Colab should ask you to authenticate yourself by logging in to your Google account. Do this, and the rest of the code will start to execute.
4. Wait for approximately 2 minutes, to allow for initial population generation, for this population to be stored in the control files, and for these files to actually update on Google Drive. Note that Google Drive is a pseudo-file-system, so writing to files by code is not actually instantaneous, hence the waiting period.
5. After waiting, navigate to Runner_1. Click the 'Runtime' dropdown, and select 'Run all'. You may have to authenticate again for this notebook, please do so. 
6. Perform step 5 above for Runner_2, Runner_3 and Runner_4, authenticating wherever needed.
7. The search has now been launched, and the runners should immediately begin evaluating their assigned members, with your patience ensuring they are getting the correctly updated files.
8. You can track the progress of the search in Controller, by scrolling down to the currently executing code block, which shows you which generation the search is on. Note that the searches done in the research take between 4 to 5 hours to execute.

**NOTE**: Keep all tabs needed for the search open, and do not disconnect from the Internet. Colab will terminate the execution of a notebook if it does not frequently receive a 'ping' from the opened notebook on your computer.

### Performance samples and comparison
1. In the My Drive/Colab Notebooks/TestSAWO folder on Google Drive, double click on the SampleComparison notebook to open it, and wait for it to load.
2. Once loaded, you can change the dataset used, as well as specify whether you want to completely regenerate the best SAWO-NN's performance sample on that dataset or just use the pre-computed performance sample. 
3. Click the 'Runtime' dropdown, and select 'Run all'. The first code block involves getting access to your Google Drive, and on your first time doing this, Colab should ask you to authenticate yourself by logging in to your Google account. Do this, and the rest of the code will start to execute.
4. Wait for the notebook to finish executing i.e for the last code block to finish. If you are regenerating, this will take about 20-25 minutes, but if not, this should take about 2 minutes.
5. Scroll down. The 3rd last code block will have printed the SAWO-NN performance sample, the 2nd last block will have printed the baseline performance sample, and the last code block will have printed the hypothesis testing result.

**NOTE**: Keep this notebook tab open, and do not disconnect from the Internet. Colab will terminate the execution of a notebook if it does not frequently receive a 'ping' from the opened notebook on your computer.

## Closing remarks
Note that Google only allows 5 Colab sessions to be active at once, which the search completely takes up. You may need to terminate one of the sessions used to search when it is completed, in order to do performance sample comparisons with the other notebook.

Feel free to fiddle with the hyperparameters of the search, and to replicate performance sample and comparison results. However, we cannot guarantee that code changed outside of the specified sections will still work effectively.

To the reviewer who requested this code upload, we hope that this repository and these instructions are to your satisfication, and thank you for your efforts in increasing the contribution of this work.

## Author information
This code and documentation was written by Jared F. O'Reilly, contributing towards his Masters degree in Computer Science at the University of Pretoria, South Africa. This code was written throughout 2021, but this repository was made available in December 2021.

