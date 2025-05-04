##@meta {desc: 'build and deployment for python projects', date: '2024-12-11'}


## Build system
#
#
# type of project
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-cli python-doc python-doc-deploy
INFO_TARGETS +=		appinfo
ADD_CLEAN +=		tmp_trainer _unsloth_temporary_saved_buffers train.log
ADD_CLEAN_ALL +=	data


## Project
#
ENTRY = 		./lmtask
GEN_PROMPT = 		"Once upon a time, in a galaxy, far far away,"


## Includes
#
include ./zenbuild/main.mk


## Targets
#
# make environment
.PHONY:			appinfo
appinfo:
			@echo "app-resources-dir: $(RESOURCES_DIR)"

# create an Python environment with Conda
.PHONY:			env
env:
			rm -rf env
			conda env create \
				--file=src/python/environment.yml \
				--prefix=env

# install Unsloth dependencies
.PHONY:			unslothdep
unslothdep:
			pip install 'unsloth==2024.8'

# stream text with the default base generation task
.PHONY:			stream
stream:
			$(ENTRY) stream base_generate 'He' \
				--override=lmtask_model_generate_args.temperature=0.5

# classify two sentences as sentiment
.PHONY:			classify
classify:
			$(ENTRY) instruct sentiment \
				'HuggingFace is a great API!\nBut the docs could improve.'

# train a new model on the tinystory corpus
.PHONY:			traintinystory
traintinystory:
			$(ENTRY) -c trainconf/tinystory.yml train

# train a new model on the databricks instruct corpus
.PHONY:			trainimdb
trainimdb:
			$(ENTRY) -c trainconf/imdb.yml train

# test the trained tiny story generation model
.PHONY:			testtinystory
testtinystory:
			$(ENTRY) -c trainconf/tinystory.yml \
				stream tinystory $(GEN_PROMPT)

# test the trained imdb instrudct model
.PHONY:			testimdb
testimdb:
			$(ENTRY) -c trainconf/imdb.yml \
				stream imdb $(GEN_PROMPT)

# run unit and integration tests
.PHONY:			testall
testall:		test stream classify
