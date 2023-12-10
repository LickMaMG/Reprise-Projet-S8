# .PHONY: help tensorboard script-create-stents script-create-dataset train

# help:
# 	@./help.sh "$(MAKEFILE_LIST)"

help:
	@echo ""


script-create-stents: ## Read .raw file and create images of 9 stents
	@python scripts/make_stents.py

script-create-dataset: ## Create dataset of noised stents ### - num : num of desc
	@python scripts/create_dataset.py \
		--basedir dataset \
		--num-images $(num)

train: ## Train a model using config files ### - cfg : config file
	@python ./src/train.py \
		--cfg $(cfg)

tensorboard: ## Run tensorboard
	tensorboard --logdir logs --port 6006

