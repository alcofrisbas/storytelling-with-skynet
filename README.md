# Storytelling with Skynet

## Authors

- Benjamin Krischer Greene
- Galen William Quinn Berger-Fletcher
- Tenzin Dophen
- Malcolm Grossman

Computer Science, Carleton College

18 March 2019

Carried out under the guise of comps.


# To Run Our Code

To make sure python dependencies are met:
```
make dep_py
```

#### <a name="train"></a>train the neural network:
```
python(3) rnn_train.py <args in cli help>
```

#### Start the webapp:
```
python(3) ./writerbot-webapp/manage.py runserver
```
RNN and SEQ2SEQ model locations are stored in the .env file, as well as other keys.

### env file

In order for the webapp to run at all, the .env file needs to be configured.

```
cp .sampleenv .env
```
First, you'll need a django secret key. You'll also need a google client id
if you want to make use of google accounts.

Now, make sure that 
1. there are trained models that the web-app can access
    * [Train a model](#train)
2. the .env points to the right directories and model names
```
#rnn model configs
PATH_TO_RNN="path/to/model"
RNN_MODEL_NAME="model_name"

PATH_TO_SEQ="path/to/model"
SEQ2SEQ_MODEL_NAME="model_name"
```
#### Please run all code from the root directory
