##@meta {desc: 'build and deployment for python projects', date: '2024-12-11'}


## Build system
#
#
# type of project
PROJ_TYPE =		python
PROJ_MODULES =		python/doc python/package python/deploy
PY_TEST_ALL_TARGETS +=	stream classify
ADD_CLEAN +=		tmp_trainer _unsloth_temporary_saved_buffers train.log
ADD_CLEAN_ALL +=	data


## Project
#
GEN_PROMPT = 		'Once upon a time, in a galaxy, far far away,'


## Includes
#
include ./zenbuild/main.mk


## Targets
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

# train a new model on the tinystory corpus
.PHONY:			traintinystory
traintinystory:
			@$(MAKE) $(PY_MAKE_ARGS) invoke \
				ARG="-c trainconf/tinystory.yml train"

## TODO
tmp:
			@CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
			$(MAKE) $(PY_MAKE_ARGS) invoke \
				ARG="-c trainconf/tinystory.yml train"

# accelerate
.PHONY:			traintinystoryacc
traintinystoryacc:	$(PY_PYPROJECT_FILE)
			$(PY_PX_BIN) run accelerate launch \
				./harness.py -c trainconf/tinystory.yml train

# train a new model on the databricks instruct corpus
.PHONY:			trainimdb
trainimdb:
			@$(MAKE) $(PY_MAKE_ARGS) invoke \
				ARG="-c trainconf/imdb.yml train"

# test the trained tiny story generation model
.PHONY:			testtinystory
testtinystory:
			@$(MAKE) $(PY_MAKE_ARGS) pyharn \
				ARG="-c trainconf/tinystory.yml \
					stream tinystory $(GEN_PROMPT)"

# test the trained imdb instrudct model
.PHONY:			testimdb
testimdb:
			@$(MAKE) ARG="-c trainconf/imdb.yml \
				stream imdb $(GEN_PROMPT)"
