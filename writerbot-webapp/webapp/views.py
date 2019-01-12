from django.shortcuts import render, redirect
from django.contrib.auth import logout as auth_logout
import sys
import os
from django.http import HttpResponse
from webapp.models import Story

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'RNN'))
#print(sys.path)
from rnn_test import load_model, generate_text
import random
from webapp.words import ADJECTIVES, ANIMALS

sess, model, word_to_id, id_to_word = None, None, None, None

# Create your views here.
def home(request):
    return render(request, 'webapp/home.html')


def write(request):
    if "prompt" not in request.session.keys():
        request.session["prompt"] = generatePrompt()

    if "editing" not in request.session.keys():
        request.session["editing"] = False

    if "sentences" not in request.session.keys():
        request.session["sentences"] = []

    global sess, model, word_to_id, id_to_word

    if not model:
        sess, model, word_to_id, id_to_word = load_model(save=True)

    suggestion = ""

    sentences = request.session.get("sentences")
    editing = request.session.get("editing")

    if request.POST:
        if request.POST.get("text"):
            newSentence = request.POST["text"]
            sentences.append(newSentence)

            if not editing:
                suggestion = generateSuggestion(newSentence)

            request.session["editing"] = not editing
    elif request.GET.get("new"):
        sentences.clear()
        request.session["editing"] = False
        request.session["prompt"] = generatePrompt(request.session.get("prompt"))

    return render(request, 'webapp/write.html',
                  context={"prompt": request.session.get("prompt"), "sentences": sentences, "suggestion": suggestion})


def about(request):
    return render(request, 'webapp/about.html')


def team(request):
    return render(request, 'webapp/team.html')


def logout(request):
    """Logs out user"""
    auth_logout(request)
    return redirect('/')


def generatePrompt(curPrompt=""):
    topics = ["a hotheaded penguin", "a wizened chihuahua", "a murderous toucan",
              "an exponentially multiplying swarm of beagles"]
    adj = ADJECTIVES[random.randrange(0, len(ADJECTIVES))]
    noun = ANIMALS[random.randrange(0, len(ANIMALS))].lower()
    curTopic = curPrompt
    while curTopic == curPrompt:
        if adj[0] in 'aeiou':
            curTopic = "Write about an {} {}".format(adj, noun)
        else:
            curTopic = "Write about a {} {}".format(adj, noun)
    return curTopic


def generateSuggestion(newSentence):
    try:
        suggestion = generate_text(sess, model, word_to_id, id_to_word, newSentence)
    except:
        suggestion = ""
    return suggestion
