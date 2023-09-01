# Prioritise-Disease-Genes
Prioritise Disease Genes Through Machine Learning method

Major library versions
-	NetworkX 2.8.8
-	node2Vec 0.4.6
-	PULearn 0.0.7
-	Scikit Learn 1.2.2
-	Pandas 1.5.3
-	Mathplotlib 3.7.1

other library and environment, please refer to requirements.txt

IMPORTANT NOTES:
1. due to the storage limitation of Github, this repository does not include the below items:
    - ./data/embeddings/
        The node embeddings of dimension 128, 96, 64 that are generated with NO random seeds
    - ./data/ppi/
        the binary files of the full ppi network graph
    - ./eval_models/
        the trained models of bagging svm (namely model_svm_1, model_svm_2)

   if not using the dataset files under ./data/datasets, but to execute all the process from end to end, 
   please note the results could not be 100% replicable, becuase new data are being downloaded and the embeddings are regenerated. 
    
   if you need the data above mentioned, please email to ronlo04@gmail.com to request a copy.

2. due to security issue, the google bigquery API key is private, you are freely to create your own from google platform and put it under ./google_bigquery_json_key

3. PULearn libary 
    - please follow the instruction to install PULearn into your vitural environment
    https://pulearn.github.io/pulearn/doc/pulearn/bagging.html
    - after installation, please replace the bagging.py in your python environment with the version located at ./updated_lib_file
    - minor changes were made to the original file in order to let it run without error.  

   Here are the minor modifications being made on the bagging.py: 
        line 46   "from sklearn.utils.metaestimators import available_if"
        line 744  "@available_if(check='base_estimator')"

This project is based on some key researches. Here are the references of this project: 

[1]	S. Picart-Armada, S. J. Barrett, D. R. Willé, A. Perera-Lluna, A. Gutteridge, and B. H. Dessailly, ‘Benchmarking network propagation methods for disease gene identification’, PLOS Comput. Biol., vol. 15, no. 9, p. e1007276, Sep. 2019, doi: 10.1371/journal.pcbi.1007276.

[2]	F. Mordelet and J.-P. Vert, ‘ProDiGe: Prioritization Of Disease Genes with multitask machine learning from positive and unlabeled examples’, BMC Bioinformatics, vol. 12, no. 1, p. 389, Dec. 2011, doi: 10.1186/1471-2105-12-389.

[3]	A. Grover and J. Leskovec, ‘node2vec: Scalable Feature Learning for Networks’. arXiv, Jul. 03, 2016. Accessed: Jun. 25, 2023. [Online]. Available: http://arxiv.org/abs/1607.00653

[4]	W. Lan, J. Wang, M. Li, W. Peng, and F. Wu, ‘Computational approaches for prioritizing candidate disease genes based on PPI networks’, Tsinghua Sci. Technol., vol. 20, no. 5, pp. 500–512, Oct. 2015, doi: 10.1109/TST.2015.7297749.

[5]	D. Ochoa et al., ‘The next-generation Open Targets Platform: reimagined, redesigned, rebuilt’, Nucleic Acids Res., vol. 51, no. D1, pp. D1353–D1359, Jan. 2023, doi: 10.1093/nar/gkac1046.

[6]	D. Szklarczyk et al., ‘STRING v11: protein–protein association networks with increased coverage, supporting functional discovery in genome-wide experimental datasets’, Nucleic Acids Res., vol. 47, no. D1, pp. D607–D613, Jan. 2019, doi: 10.1093/nar/gky1131.

[7]	R. Wright, ‘Positive-unlabeled learning – Roy Wright’.

[8]	F. Pedregosa et al., ‘Scikit-learn: Machine Learning in Python’, Mach. Learn. PYTHON.

[9]	W. McKinney, ‘Data Structures for Statistical Computing in Python’, presented at the Python in Science Conference, Austin, Texas, 2010, pp. 56–61. doi: 10.25080/Majora-92bf1922-00a.

[10]	J. D. Hunter, ‘Matplotlib: A 2D graphics environment’, Comput. Sci. Eng., vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.