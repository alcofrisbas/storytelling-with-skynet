from django.shortcuts import render, redirect
from django.contrib.auth import logout as auth_logout

from django.http import HttpResponse
from wsgiref.util import FileWrapper
from io import StringIO

from webapp.models import Story
from webapp.models import User

import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'simpleRNN'))
from rnn_words import SimpleRNN
import tensorflow as tf
import random
from webapp.words import ADJECTIVES, ANIMALS

# little hacky shit to make pickling work for loading the ngram model
# fastly.
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'./'))
from ngrams import ngram
sys.modules["ngram"] = ngram


from enum import Enum

class Mode(Enum):
     RNN = 1
     NGRAM = 2
     NONE = 3

args_dict = {"n_input": 4, "batch_size": 1, "n_hidden": 300, "learning_rate": 0.001, "training_iters": 50000, "training_file": "simpleRNN/data/train.txt"}
display_step = 1000
path_to_model = "simpleRNN/models/"
model_name = "basic_model"

print("instantiating RNN")
rnn = SimpleRNN(args_dict, display_step, path_to_model, model_name)
print("instantiating saver")
sess = tf.Session()
saver = tf.train.Saver()
print("loading saved RNN from " + rnn.path_to_model)
saver.restore(sess, tf.train.latest_checkpoint(rnn.path_to_model))

print("loading saved ngram")
ngram_model = ngram.NGRAM_model("./saves/all_of_lewis.txt", "lewis_model", "./ngrams/models")
prompt_model = ngram.NGRAM_model("./saves/all_of_lewis.txt", "lewis_model", "./ngrams/models")

ngram_model.m = 2
ngram_model.high = 100
prompt_model.m = 2

if not os.path.isfile("./ngrams/models/lewis_model"):
    prompt_model.create_model("./saves/all_of_lewis.txt","./ngrams/models/lewis_model")
    prompt_model.create_model("./saves/all_of_lewis.txt","./ngrams/models/lewis_model")
else:
    ngram_model.load_model()
    prompt_model.load_model()

#TODO: when user logs in, redirect to the page they logged in from
#TODO: figure out how to clear empty stories and expired session data

# Create your views here.
def home(request):
    if request.user.is_authenticated:
        user = getOrCreateUser(request)
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
    user, new = User.objects.get_or_create(email=request.user.email)
    if new:
        user.first_name = request.user.first_name
        user.last_name = request.user.last_name
        user.save()
    return user


def newStory(request):
    if request.session.get("story_id") and not request.user.is_authenticated:
        old_story = Story.objects.get(id=request.session.get("story_id"))
        old_story.delete()

    s = Story.objects.create(sentences="", title="Untitled", prompt=generatePrompt())
    if request.user.is_authenticated:
        user = getOrCreateUser(request)
        s.author = user
        s.save()
        user.stories.add(s)
        user.save()
    request.session["story_id"] = s.id
    return redirect('/write')


#TODO: error check
def loadStory(request, id):
    if Story.objects.filter(id=id).exists():
        request.session["story_id"] = id
        return redirect('/write')
    else:
        return render(request, 'webapp/error.html', context= {'message': "Story not found."})


def deleteStory(request, id):
    if Story.objects.filter(id=id).exists():
        s = Story.objects.get(id=id)
        if id == request.session.get("story_id"):
            print("deleting request.session {}".format(id))
            request.session.pop("story_id")
        if s.author.email == request.user.email:
            s.delete()
            return redirect('/')
        else:
            return render(request, 'webapp/error.html',
                          context={'message': "Sorry, you don't have permission to access that story. Try logging in."})
    else:
        return render(request, 'webapp/error.html', context={'message': "Story not found."})


def write(request):
    if "story_id" not in request.session.keys() or not Story.objects.filter(id = request.session["story_id"]).exists():
        newStory(request)

    if "mode" not in request.session.keys():
        # using value instead of enum itself because enum is not JSON serializable so it can't be stored in session
        request.session["mode"] = Mode.RNN.value

    story = Story.objects.get(id=request.session["story_id"])
    suggestion = ""

    if request.POST:
        if request.POST.get("text"):
            new_sentence = request.POST["text"]
            story.sentences += new_sentence.strip().replace("\n", "") + "\n"
            story.suggesting = not story.suggesting
            story.save()
            if request.session.get("mode") != Mode.NONE.value and story.suggesting:
                suggestion = generateSuggestion(sess, new_sentence, request.session.get("mode"))

        if request.POST.get("title"):
            story.title = request.POST["title"]
            story.save()

        if request.POST.get("re-prompt"):
            story.prompt = generatePrompt()
            story.save()

        # same functionality as "Start a new story" button
        if request.POST.get("new"):
            return redirect('/new_story')

        if request.POST.get("home"):
            return redirect('/')

        if request.POST.get("export"):
            print("exporting story...")
            myfile = StringIO()
            myfile.write(story.sentences)
            response = HttpResponse(myfile.getvalue(), content_type='text/plain')
            response['Content-Disposition'] = 'attachment; filename={}.txt'.format(story.title)
            return response

        if request.POST.get('mode') == "rnn_mode":
            request.session["mode"] = Mode.RNN.value
        elif request.POST.get('mode') == "ngram_mode":
            request.session["mode"] = Mode.NGRAM.value
        elif request.POST.get('mode') == "none_mode":
            request.session["mode"] = Mode.NONE.value

        if request.POST.get('prompt_mode') == "no_prompt":
            story.prompt = ""
            story.save()
        elif story.prompt == "":
            story.prompt = generatePrompt()
            story.save()

    elif request.GET.get("new"):
        return redirect('/new_story')
    else:
        if request.session.get("mode") != Mode.NONE.value and story.suggesting and story.sentences != "":
            last = story.sentences.split("\n")[-2]
            suggestion = generateSuggestion(sess, last, request.session.get("mode"))

    return render(request, 'webapp/write.html',
                  context={"prompt": story.prompt,
                  "sentences": [s.strip() for s in story.sentences.split("\n")[:-1]],
                  "suggestion": suggestion, "title": story.title,
                  "mode": request.session["mode"]})


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


def generatePrompt(curPrompt=""):
    # adj = ADJECTIVES[random.randrange(0, len(ADJECTIVES))]
    # noun = ANIMALS[random.randrange(0, len(ANIMALS))].lower()
    # curTopic = curPrompt
    # while curTopic == curPrompt:
    #     if adj[0] in 'aeiou':
    #         curTopic = "Write about an {} {}".format(adj, noun)
    #     else:
    #         curTopic = "Write about a {} {}".format(adj, noun)
    curTopic = prompt_model.generate_with_constraints("STOP")
    return curTopic


def generateSuggestion(session, newSentence, mode):
    try:
        if mode == Mode.RNN.value:
            suggestion = rnn.generate_suggestion(session, newSentence)
        elif mode == Mode.NGRAM.value:
            suggestion = ngram_model.generate_with_constraints(newSentence)
        else:
            suggestion="placeholder"
    except Exception as e:
        print("ERROR (suggestion generation)")
        suggestion = e
    return suggestion
