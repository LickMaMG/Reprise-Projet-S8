help:
	@echo "Commands:"
	@echo
	@echo "\033[1;33mconnect\033[0m                          	     <-- Connect to the server."
	@echo "-------------------------------------"
	@echo "\033[1;33minit-venv\033[0m                      	     <-- Create a python virtual environment named venv."
	@echo "\033[1;33mdelete-venv\033[0m                    	     <-- Delete virtual environment."
	@echo "-------------------------------------"
	@echo "\033[1;33minit-docker-emotions\033[0m           	     <-- Create docker image."
	@echo "\033[1;33mlaunch-docker-emotions-gpu\033[0m     	     <-- Launch docker environment."
	@echo "-------------------------------------"
	@echo "\033[1;33mtensorboard-server\033[0m              	     <-- Make the link to server tensorboard."
	@echo "\033[1;33mtensorboard\033[0m                    	     <-- Launch tensorboard. Don't forget to activate virtual environment."
	@echo "-------------------------------------"
	@echo "\033[1;33mgenerate-dataset\033[0m <-- Extract frames of each video into 256x256 shape and create description files"
	@echo "\033[1;33mcount-aus\033[0m <-- Create txt files of AUs' combinations parts for each macro-emotion "
	@echo "-------------------------------------"
	@echo "\033[1;33mvisualize_random_video\033[0m         	     <-- Visualize random video from the dataset."
	@echo "\033[1;33mvisualize_video\033[0m                	     <-- Visualize video whose path is given after the command."
	@echo "\033[1;33mvisualize_random_sequence\033[0m      	     <-- Visualize random sequence."
	@echo "\033[1;33mvisualize_sequence\033[0m                   <-- ..."
	@echo "-------------------------------------"
	@echo "\033[1;33mtrain-slowfast\033[0m        <-- Train slowfast model with resnet50 as backbone for 32 frames by sequence. Training is launch with config files."
	

#! rajouter les intructions

hello:
	@mkdir hello

# Conda environments

# create:
# 	@conda env create --prefix ./env --file tf.yml

# activate-env:
# 	@source /home/malick/miniconda3/condabin/conda/bin/activate /home/malick/Desktop/kss/env

# remove-env:
# 	@conda remove --prefix ./env --all -y

# Tensorboard
tensorboard:
	tensorboard --logdir logs --port 6006

# all scripts

script-make-stents:
	@python scripts/make_stents.py

# Train

train:
	@python ./src/train.py \
	--cfg $(cfg)
	