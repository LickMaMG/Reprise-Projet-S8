help:
	@echo "Commands:"
	@echo




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

script-create-stents:
	@python scripts/make_stents.py

script-create-dataset:
	@python scripts/create_dataset.py \
	--basedir dataset \
	--num-images $(num)

# Train

train:
	@python ./src/train.py \
	--cfg $(cfg)
	