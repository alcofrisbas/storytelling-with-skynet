from django.shortcuts import render, redirect
from django.contrib.auth import logout as auth_logout
import sys
import os
from django.http import HttpResponse
from webapp.models import Story
from webapp.models import User

#sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'RNN'))
#from rnn_test import load_model, generate_text
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'simpleRNN'))
#from rnn_words import SimpleRNN
#from rnn_words2 import run
import random
from webapp.words import ADJECTIVES, ANIMALS


args_dict = {"n_input": 4, "batch_size": 2, "n_hidden": 300, "learning_rate": 0.001, "training_iters": 50000}
display_step = 1000
path_to_model = "RNN/models/"
model_name = "best_model"
#rnn = SimpleRNN(args_dict, display_step, path_to_model, model_name)

#TODO: when user logs in, redirect to the page they logged in from
#TODO: figure out how to clear empty stories and expired session data

# Create your views here.
def home(request):
    if request.user.is_authenticated:
        user = getOrCreateUser(request)
        if request.session.get("story_id"):
            cur_story = Story.objects.get(id=request.session.get("story_id"))
            cur_story.author = user
            cur_story.save()
            user.stories.add(cur_story)
            user.save()
        stories = user.stories.all()
    else:
        stories = []
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
    request.session["editing"] = False
    request.session["story_id"] = s.id
    return redirect('/write')


#TODO: figure out how editing/prompt interact with story loading
#TODO: error check
def loadStory(request, id):
    if Story.objects.filter(id=id).exists():
        request.session["story_id"] = id
        request.session["editing"] = False
        request.session["prompt"] = ""
        return redirect('/write')
    else:
        return render(request, 'webapp/error.html', context= {'message': "Story not found."})


def deleteStory(request, id):
    if Story.objects.filter(id=id).exists():
        s = Story.objects.get(id=id)
        if s.author.email == request.user.email:
            s.delete()
            if id == request.session.get("story_id"):
                request.session.pop("story_id")
            return redirect('/')
        else:
            return render(request, 'webapp/error.html',
                          context={'message': "Sorry, you don't have permission to access that story. Try logging in."})
    else:
        return render(request, 'webapp/error.html', context={'message': "Story not found."})


def write(request):
    if "story_id" not in request.session.keys() or not Story.objects.filter(id = request.session["story_id"]).exists():
        newStory(request)

    if "AI" not in request.session.keys():
        request.session["AI"] = True

    # if "sess" not in request.session.keys():
    #     request.session["sess"] = rnn.load()
    #sess = rnn.load()

    story = Story.objects.get(id=request.session["story_id"])
    suggestion = ""

    if request.POST:
        if request.POST.get("text"):
            new_sentence = request.POST["text"]
            story.sentences += new_sentence.strip() + "\n"
            story.save()

            if request.session["AI"] and not request.session["editing"]:
                suggestion = generateSuggestion(None, new_sentence)

            request.session["editing"] = not request.session["editing"]

        if request.POST.get("title"):
            story.title = request.POST["title"]
            story.save()

        # same functionality as "Start a new story button"
        if request.POST.get("new"):
            return redirect('/new_story')

        if request.POST.get("home"):
            return redirect('/')

        if request.POST.get("side-toggle"):
            request.session["AI"] = not request.session["AI"]

    elif request.GET.get("new"):
        return redirect('/new_story')

    last = ""
    if story.sentences != "":
        last = story.sentences.split("\n")[-1]

    return render(request, 'webapp/write.html',
                  context={"prompt": story.prompt,
                  "sentences": [s.strip() for s in story.sentences.split("\n")[:-1]],
                  "suggestion": suggestion, "last": last,
                  "title": story.title, "AI": request.session["AI"]})


def about(request):
    return render(request, 'webapp/about.html')


def team(request):
    return render(request, 'webapp/team.html')


def error(request, message):
    return render(request, 'webapp/error.html', context={'message': message})


def saves(request):
    stories = Story.objects.all()
    return render(request, 'webapp/saves.html', context={'stories': stories})


def logout(request):
    """Logs out user"""
    auth_logout(request)
    return redirect('/')


def generatePrompt(curPrompt=""):
    adj = ADJECTIVES[random.randrange(0, len(ADJECTIVES))]
    noun = ANIMALS[random.randrange(0, len(ANIMALS))].lower()
    curTopic = curPrompt
    while curTopic == curPrompt:
        if adj[0] in 'aeiou':
            curTopic = "Write about an {} {}".format(adj, noun)
        else:
            curTopic = "Write about a {} {}".format(adj, noun)
    return curTopic


def generateSuggestion(session, newSentence):
    try:
        #suggestion = generate_text(sess, model, word_to_id, id_to_word, seed=newSentence)
        #suggestion = rnn.generate_suggestion(session, newSentence)
        suggestion="placeholder"
    except Exception as e:
        print("ERROR (suggestion generation)")
        suggestion = e
    return suggestion
