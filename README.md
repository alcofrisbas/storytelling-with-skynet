# comps2018-19

Galen
Malcolm
Ben
Tenzin

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
Now, make sure that 
1. there are trained models that the web-app can access
Take me to [pookie](#train)
2. the .env points to the right directories and model names 
#### Please run all code from the root directory
