from django.shortcuts import render, redirect
from django.contrib.auth import logout as auth_logout

from django.http import HttpResponse
from wsgiref.util import FileWrapper
from io import StringIO

from webapp.models import Story
from webapp.models import User
import os

from decouple import config



import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'simpleRNN'))
from rnn_words import SimpleRNN
from rnn_words_seq2seq import SimpleRNN as seq2seqRNN
import tensorflow as tf
import random
from webapp.words import ADJECTIVES, ANIMALS

# little hacky shit to make pickling work for loading the ngram model fastly.
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'./'))
from ngrams import ngram
sys.modules["ngram"] = ngram


from enum import Enum

# an enum representing the algorithm used to generate suggestions. Values used
# in write page and generateSuggestion -- if you add any values, add handlers for
# them there
class Mode(Enum):
     RNN = 0
     SEQ2SEQ = 1
     NGRAM = 2
     NONE = 3

# an enum representing the algorithm used to generate prompts. Values used
# in write page and generatePrompt -- if you add any values, add handlers for
# them there
class PromptMode(Enum):
    SIMPLE = 0
    LEWIS = 1
    DICKENS = 2
    NONE = 3

args_dict = {"n_input": 20, "batch_size": 1, "n_hidden": 300, "learning_rate": 0.001, "training_iters": 50000, "training_file": "simpleRNN/data/charles_dickens_great_expectations1.txt"}
display_step = 1000
path_to_model = config("PATH_TO_RNN")#"simpleRNN/models/"
path_to_seq2seq_model = config("PATH_TO_SEQ")#"simpleRNN/seq2seq_models/"
model_name = config("RNN_MODEL_NAME")#"basic_model"
seq2seq_model_name = config("SEQ2SEQ_MODEL_NAME")#"seq2seq_model_name"
seq2seq_args_dict = {"n_input": 6, "batch_size": 1, "n_hidden": 500, "learning_rate": 0.001, "training_iters":50000, "training_file": "simpleRNN/data/charles_dickens_great_expectations1.txt"}

print("instantiating RNN")
sess = tf.Session()
rnn = SimpleRNN(args_dict, display_step, path_to_model, model_name)
print("instantiating saver")
saver = tf.train.Saver()
print("loading saved RNN from " + rnn.path_to_model)
saver.restore(sess, tf.train.latest_checkpoint(rnn.path_to_model))

with tf.Graph().as_default():
    seq2seq_rnn = seq2seqRNN(seq2seq_args_dict, display_step, path_to_seq2seq_model, seq2seq_model_name, False)
    seq2seq_sess = tf.Session()
    #with tf.variable_scope("seq2seq"):
    print("loading saved seq2seqRNN from " + seq2seq_rnn.path_to_model)
    seq2seq_saver = tf.train.Saver()
    seq2seq_saver.restore(seq2seq_sess, tf.train.latest_checkpoint(seq2seq_rnn.path_to_model))

print("loading saved ngram")
ngram_model = ngram.NGRAM_model("./ngrams/models")
prompt_model = ngram.NGRAM_model("./ngrams/models")
ngram_model.create_model("5max200000.model")
ngram_model.set_model("5max200000.model")

prompt_model.create_model("5max200000.model")
#prompt_model.create_model("lewis_model2")
prompt_model.set_model("5max200000.model")
ngram_model.m = 2
ngram_model.high = 100
prompt_model.m = 2


#TODO: clear empty stories and expired session data with a cron job
def home(request):
    # if user is logged in, pass their stories as context. otherwise just use the current story
    if request.user.is_authenticated:
        user = getOrCreateUser(request)
        # assign the current story to the user in case they started it while not logged in
        if request.session.get("story_id"):
            try:
                cur_story = Story.objects.get(id=request.session.get("story_id"))
                cur_story.author = user
                cur_story.save()
                user.stories.add(cur_story)
                user.save()
            except Exception as e:
                print(e)
                print("id: {}".format(request.session.get("story_id")))
        stories = user.stories.all()
    else:
        stories = []
        if request.session.get("story_id"):
            stories.append(Story.objects.get(id=request.session.get("story_id")))

    return render(request, 'webapp/home.html', context={'stories': stories})


def getOrCreateUser(request):
    """
    Retrieves the entry associated with the current user if logged in.
    If no such entry is found, creates one with the current user's information.
    If user is not logged in, returns None.
    """
    if request.user.is_authenticated:
        user, new = User.objects.get_or_create(email=request.user.email)
        if new:
            user.first_name = request.user.first_name
            user.last_name = request.user.last_name
            user.save()
        return user
    else:
        return None


def newStory(request):
    """
    Creates a new story with default settings, assigned to the current user if logged in.
    """
    # clears the user's old story, if any, if they aren't logged in
    if request.session.get("story_id") and not request.user.is_authenticated:
        old_story = Story.objects.get(id=request.session.get("story_id"))
        old_story.delete()

    s = Story.objects.create(sentences="", title="Untitled",
        prompt=generatePrompt(0))
    if request.user.is_authenticated:
        user = getOrCreateUser(request)
        s.author = user
        s.save()
        user.stories.add(s)
        user.save()
    request.session["story_id"] = s.id
    return redirect('/write')


def loadStory(request, id):
    """
    Loads the story identified by id unless it belongs to a different user, in which case
    returns an error.
    Returns an error if the story doesn't exist.
    """
    if Story.objects.filter(id=id).exists():
        s = Story.objects.get(id=id)
        if (not s.author.is_authenticated) or s.author == request.user:
            request.session["story_id"] = id
            return redirect('/write')
        else:
            return render(request, 'webapp/error.html',
                          context={'message': "Sorry, you don't have permission to access that story. Try logging in."})
    else:
        return render(request, 'webapp/error.html', context= {'message': "Story not found."})


def deleteStory(request, id):
    """
    Deletes the story identified by id if it belongs to the current user, otherwise
    returns an error.
    """
    if Story.objects.filter(id=id).exists():
        s = Story.objects.get(id=id)
        if request.user.is_authenticated:
            if s.author.email == request.user.email:
                if id == request.session.get("story_id"):
                    print("deleting request.session {}".format(id))
                    request.session.pop("story_id")
                s.delete()
                return redirect('/')
            else:
                return render(request, 'webapp/error.html',
                                  context={'message': "Sorry, you don't have permission to delete that story."})
        else:
            return render(request, 'webapp/error.html',
                              context={'message': "Sorry, you don't have permission to access that story. Try logging in."})
    else:
        return render(request, 'webapp/error.html', context={'message': "Story not found."})


def write(request):
    """
    Handles logic for the primary write page.
    """
    if "story_id" not in request.session.keys() or not Story.objects.filter(id = request.session["story_id"]).exists():
        newStory(request)

    story = Story.objects.get(id=request.session["story_id"])
    suggestion = ""

    if request.POST:
        if request.POST.get("text"):
            new_sentence = request.POST["text"]
            story.sentences += new_sentence.strip().replace("\n", "") + "\n"
            story.suggesting = not story.suggesting
            story.save()
            if story.mode != Mode.NONE.value and story.suggesting:
                if story.mode == Mode.RNN.value or story.mode == Mode.NGRAM.value:
                    suggestion = generateSuggestion(sess, new_sentence, story.mode)
                elif story.mode == Mode.SEQ2SEQ.value:
                    suggestion = generateSuggestion(seq2seq_sess, new_sentence, story.mode)
                else:
                    print("Unimplemented mode {}".format(story.mode))

        if request.POST.get("title"):
            story.title = request.POST["title"]
            story.save()

        if request.POST.get("re-prompt"):
            story.prompt = generatePrompt(story.prompt_mode)
            story.save()

        if request.POST.get("new"):
            return redirect('/new_story')

        if request.POST.get("home"):
            return redirect('/')

        if request.POST.get("export"):
            myfile = StringIO()
            myfile.write(story.sentences)
            response = HttpResponse(myfile.getvalue(), content_type='text/plain')
            response['Content-Disposition'] = 'attachment; filename={}.txt'.format(story.title)
            return response

        if request.POST.get('mode'):
            story.mode = request.POST['mode']
            story.save()

        if request.POST.get('prompt_mode') and int(story.prompt_mode) != int(request.POST['prompt_mode']):
            story.prompt_mode = request.POST['prompt_mode']
            story.prompt = generatePrompt(story.prompt_mode)
            story.save()

    elif request.GET.get("new"):
        return redirect('/new_story')
    else:
        if story.mode != Mode.NONE.value and story.suggesting and story.sentences != "":
            last = story.sentences.split("\n")[-2]
            if story.mode == Mode.RNN.value or story.mode == Mode.NGRAM.value:
                suggestion = generateSuggestion(sess, last, story.mode)
            elif story.mode == Mode.SEQ2SEQ.value:
                suggestion = generateSuggestion(seq2seq_sess, last, story.mode)
            else:
                print("Unimplemented mode {}".format(story.mode))

    return render(request, 'webapp/write.html',
                  context={"story": story,
                  "sentences": [s.strip() for s in story.sentences.split("\n")[:-1]],
                  "suggestion": suggestion,
                  "modes": [(mode.name, mode.value) for mode in Mode],
                  "prompt_modes": [(mode.name, mode.value) for mode in PromptMode]})


def about(request):
    return render(request, 'webapp/about.html')


def team(request):
    return render(request, 'webapp/team.html')


def error(request, message):
    return render(request, 'webapp/error.html', context={'message': message})


def logout(request):
    """Logs out user"""
    auth_logout(request)
    return redirect('/')


def generatePrompt(prompt_mode):
    """Generates a prompt according to prompt_mode, which should be a
    value of the PromptMode enum."""
    if int(prompt_mode) == PromptMode.SIMPLE.value:
        adj = ADJECTIVES[random.randrange(0, len(ADJECTIVES))]
        noun = ANIMALS[random.randrange(0, len(ANIMALS))].lower()
        if adj[0] in 'aeiou':
            prompt = "Write about an {} {}".format(adj, noun)
        else:
            prompt = "Write about a {} {}".format(adj, noun)
    elif int(prompt_mode) == PromptMode.LEWIS.value:
        prompt_model.set_model("lewis_model2")
        prompt = prompt_model.generate_with_constraints("STOP")
    elif int(prompt_mode) == PromptMode.DICKENS.value:
        prompt_model.set_model("dickens_model")
        prompt = prompt_model.generate_with_constraints("STOP")
    elif int(prompt_mode) == PromptMode.NONE.value:
        prompt = ""
    else:
        print("unknown prompt mode {}".format(prompt_mode))
        prompt = ""
    return prompt


def generateSuggestion(session, newSentence, mode):
    """Generates a suggestion sentence based on a tensorflow session, the previous
    sentence, and a value of the Mode enum."""
    try:
        if int(mode) == Mode.RNN.value:
            suggestion = rnn.generate_suggestion(session, newSentence)
        elif int(mode) == Mode.SEQ2SEQ.value:
            suggestion = seq2seq_rnn.generate_suggestion(session, newSentence)
        elif int(mode) == Mode.NGRAM.value:
            suggestion = ngram_model.generate_with_constraints(newSentence)
        elif int(mode) == Mode.NONE.value:
            suggestion = ""
        else:
            print("unknown mode {}".format(mode))
            suggestion = ""
    except Exception as e:
        print("ERROR (suggestion generation)")
        suggestion = e
    return suggestion
