# Ngram models for text prediction

## Usage
``` python
# create a model
model = NGRAM_model("path/to/model/location")
# warn: if the model does not already exist,
# not having data will raise an exception
model.create_model(model_name, [data=data_file_name, max_read=amount_to_read])
# set which model to use
# this enables the model to be hot-swappable
model.set_model(model_name)

model.generate_with_constraints("input sentence.")
```

### Further customizing of the ngram model:
```python
model.depth = 5 # how deep to make the Trie -- max ngram query
model.l = 100000 how many words to train on
model.display_step = 2000 # how often to print during training
model.m = 3 # max depth of sentence-gen n-gram
model.sent_length = 200 # max length of sentence gen in the
                        # event that a STOP is not reached
model.low = 10 # lower bound on  naturally terminated sentences
model.high = 75 # upper bound on naturally terminated sentences
```
