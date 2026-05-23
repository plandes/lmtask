##@meta {desc: 'build and deployment for python projects', date: '2024-12-11'}


## Build system
#
#
# type of project
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/package python/deploy
PY_TEST_ALL_TARGETS +=	stream classify
ADD_CLEAN +=		tmp_trainer train.log
ADD_CLEAN_ALL +=	data


## Project
#
TEST_MODEL ?=		llama
CONFIG ?=		trainconf/$(TASK)-$(TEST_MODEL).yml
GEN_PROMPT ?= 		'Once upon a time, in a galaxy, far far away,'


## Includes
#
include ./zenbuild/main.mk


## Inference targets
#
# stream text with the default base generation task
.PHONY:			stream
stream:
			@$(MAKE) $(PY_MAKE_ARGS) invoke \
				ARG="stream base_generate 'He' \
				--override=lmtask_model_generate_args.temperature=0.5"

# classify two sentences as sentiment
.PHONY:			classify
classify:
			@$(MAKE) $(PY_MAKE_ARGS) pyharn ARG="instruct sentiment \
				'HuggingFace is a great API!\nBut the docs could improve.'"

## Train
#
# train a new model
.PHONY:			train
train:			$(PY_PYPROJECT_FILE)
			@$(MAKE) $(PY_MAKE_ARGS) invoke ARG="-c $(CONFIG) train"

# accelerate
.PHONY:			trainacc
trainacc:		$(PY_PYPROJECT_FILE)
			$(PY_PX_BIN) run accelerate launch \
				./harness.py -c $(CONFIG) train

# train a new qwen model on the tinystory corpus
.PHONY:			traintinystoryqwen
traintinystoryqwen:
			@$(MAKE) $(PY_MAKE_ARGS) \
				TASK=tinystory TEST_MODEL=qwen3 train

# train a new gemma model on the tinystory corpus
.PHONY:			traintinystorygemma
traintinystorygemma:
			@$(MAKE) $(PY_MAKE_ARGS) \
				TASK=tinystory TEST_MODEL=gemma4 train

# train a new llama3 model on the databricks instruct corpus
.PHONY:			trainimdbllama3
trainimdbllama3:
			@$(MAKE) $(PY_MAKE_ARGS) \
				TASK=imdb TEST_MODEL=llama3 train

# train a new llama3 model on the databricks instruct corpus
.PHONY:			trainimdbqwen3
trainimdbqwen3:
			@$(MAKE) $(PY_MAKE_ARGS) \
				TASK=imdb TEST_MODEL=qwen3 train

# train a new llama3 model on the databricks instruct corpus
.PHONY:			trainimdbgemma4
trainimdbgemma4:
			@$(MAKE) $(PY_MAKE_ARGS) \
				TASK=imdb TEST_MODEL=gemma4 train

# retrain all imdb task models
.PHONY:			retrainimdb
retrainimdb:
			rm -fr data/imdb
			for model in llama3 qwen3 gemma4 ; do \
				make TASK=imdb TEST_MODEL=$$model train ; \
			done


## Test
#
# test the trained tiny story generation model
.PHONY:			testtinystory
testtinystory:
			@$(MAKE) $(PY_MAKE_ARGS) pyharn \
				ARG="-c $(CONFIG) stream tinystory $(GEN_PROMPT)"

# test the trained imdb instrudct model
.PHONY:			testimdb
testimdb:
			@$(MAKE) $(PY_MAKE_ARGS) pyharn \
				ARG="-c trainconf/imdb.yml \
					instruct imdb 'I loved the movie'"
