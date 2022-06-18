# Incremental Data-Uploading for Full-Quantum Classification

A Quantum Classifier for MNIST dataset. This is the code accompanying the paper: https://doi.org/10.48550/arXiv.2205.03057

To run the file, follow the following steps:

1.  Convert the dataset into numpy array and save it as npy file. Example can be seen in  create_npy.py
2.  Run the Quantum_MNIST_pkl_DRU.py file with the required command line arguments. For example,

    python -u Quantum_MNIST_pkl_DRU.py --log_dir=ComplexEnc/ --enc_layers=8 --arch=localized_encoding_multi

3.  For calculating the effective dimension, use effective_dimension_calculate.py file.

    The effective dimension implementation is adapted from the paper: The power of quantum neural networks (https://doi.org/10.1038/s43588-021-00084-1)

