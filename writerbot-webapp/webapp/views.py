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

    if "newStory" not in request.session.keys():
        request.session["newStory"] = True

    global sess, model, word_to_id, id_to_word

    if not model:
        sess, model, word_to_id, id_to_word = load_model(save=False)

    suggestion = ""
    sentences = request.session.get("sentences")
    editing = request.session.get("editing")
    title = request.session.get("title")
    newStory = request.session.get("newStory")

    if request.POST:
        if request.POST.get("text"):
            newSentence = request.POST["text"]
            sentences.append(newSentence)

            if not editing:
                suggestion = generateSuggestion(newSentence)

            request.session["editing"] = not editing

        if request.POST.get("title"):
            title = request.POST["title"]
            request.session["title"] = title

        # STILL WORKING THIS
        if request.POST.get("save"):
            print("saving")
            title = request.POST["title"]
            request.session["title"] = title
            if request.session.get("newStory"):
                print("making new Story")
                Story.objects.create(sentences = title, title=sentences)
                request.session["newStory"] = False
            else:
                print(Story.objects.all().filter(title=title))
                s = Story.objects.all().filter(title=title)
                print(s)


    elif request.GET.get("new"):
        sentences.clear()
        request.session["editing"] = False
        request.session["prompt"] = generatePrompt(request.session.get("prompt"))
        request.session["newStory"] = True

    # elif request.GET.get("save"):
    #     #title = request.POST["title"]
    #     print(title)
    #     print(sentences)
    #     Story.objects.create(sentences = "\n".join(sentences), title=title)
        #print(Story.objects.all())


    last = ""
    if sentences:
        last = sentences[-1]
        #sentences.pop()
    return render(request, 'webapp/write.html',
                  context={"prompt": request.session.get("prompt"), "sentences": sentences[:-1], "suggestion": suggestion, "last":last})


def about(request):
    return render(request, 'webapp/about.html')


def team(request):
    return render(request, 'webapp/team.html')

def saves(request):
    # need to implement the title and author thing....
    # author needs users/auth
    stories = Story.objects.all()
    #stories = [s.sentences for s in stories]
    return render(request, 'webapp/saves.html',context={'stories': stories})

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


def generateSuggestion(newSentence):
    try:
        suggestion = generate_text(sess, model, word_to_id, id_to_word, seed=newSentence)
    except Exception as e:
        print("ERROR (suggestion generation)")
        suggestion = e
    return suggestion
