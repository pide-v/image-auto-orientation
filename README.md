# image-auto-orientation

This repo contains code for defining and training models to classify images that are correctly oriented. 

At the root folder of the project this python files can be found:

	- datageneration.py: It exploits some utilities defined in code/utils.py in order to generate the augmented dataset for this task, starting from any dataset containing images.

	-gputest.py: It's a quick way to test if your system can use CUDA to train the models on GPU

	-modelrunner.py: It is a simple GUI that allows the loading of .h5 saved models and also the querying of such models.

Root /code contains the definition and the training scripts for all the models

	/code/utils.py Contains some utility functions (For generating the datasets and saving training results and models)

	/code/mnist 
	/code/cifar10

	/code/simplecnn
	/code/transfer-learning

	/code/street-view