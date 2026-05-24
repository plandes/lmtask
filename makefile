##@meta {desc: 'build and deployment for python projects', date: '2024-12-11'}


## Build system
#
#
# type of project
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/package python/deploy
PY_TEST_ALL_TARGETS +=	trainimdb stream classify
ADD_CLEAN +=		tmp_trainer train.log
ADD_CLEAN_ALL +=	data


## Project
#
MODELS ?=		llama3 qwen3 gemma4
TRAIN_CONF_DIR ?=	trainconf
CONFIG ?=		$(TRAIN_CONF_DIR)/$(TASK)-$(TEST_MODEL).yml
GEN_PROMPT ?= 		'Once upon a time, in a galaxy, far far away,'


## Includes
#
include ./zenbuild/main.mk


## Train and inference function-like
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

# train a new model
.PHONY:			train
train:			$(PY_PYPROJECT_FILE)
			@if [ -d data/$(TASK)/$(TEST_MODEL) ] ; then \
				echo "$(TASK) $(TEST_MODEL) already exists" ; \
			else \
				echo "training $(TASK) $(TEST_MODEL)" ; \
				$(MAKE) $(PY_MAKE_ARGS) invoke \
					ARG="-c $(CONFIG) train" ; \
			fi


## Tinystory train
#
# train all tinystory task models
.PHONY:			traintinystory
traintinystory:
			@for model in $(MODELS) ; do \
				$(MAKE) $(PY_MAKE_ARGS) \
					TASK=tinystory TEST_MODEL=$$model train ; \
			done

# retrain all tinystory task models
.PHONY:			retraintinystory
retraintinystory:
			rm -fr data/tinystory
			@$(MAKE) $(PY_MAKE_ARGS) traintinystory


## IMDB train
#
# train all imdb task models
.PHONY:			trainimdb
trainimdb:
			@for model in $(MODELS) ; do \
				$(MAKE) $(PY_MAKE_ARGS) \
					TASK=imdb TEST_MODEL=$$model train ; \
			done

# retrain all imdb task models
.PHONY:			retrainimdb
retrainimdb:
			rm -fr data/imdb
			@$(MAKE) $(PY_MAKE_ARGS) trainimdb


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
				ARG="-c $(TRAIN_CONF_DIR)/imdb-llama3.yml \
					instruct dataset 'I loved the movie'"
