from django.shortcuts import render, redirect
from django.contrib.auth import logout as auth_logout
import sys
import os
from django.http import HttpResponse
from webapp.models import Story
from webapp.models import User

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'RNN'))
from rnn_test import load_model, generate_text
import random
from webapp.words import ADJECTIVES, ANIMALS

sess, model, word_to_id, id_to_word = None, None, None, None

# Create your views here.
def home(request):
    #stories = Story.objects.all()
    try:
        user = getOrCreateUser(request)
        stories = user.stories.all()
    except Exception as e:
        print(e)
        stories = []
    return render(request, 'webapp/home.html', context={'stories': stories})


def getOrCreateUser(request):
    user, new = User.objects.get_or_create(email=request.user.email)
    user.first_name = request.user.first_name
    user.last_name = request.user.last_name
    user.save()
    return user


def newStory(request):
    request.session["sentences"] = ""
    request.session["title"] = ""
    request.session["editing"] = False
    request.session["prompt"] = generatePrompt(request.session.get("prompt"))
    request.session["newStory"] = True
    return redirect('/write')


#TODO: figure out how editing interacts with story loading
def loadStory(request, title):
    request.session["title"] = title
    s = Story.objects.get(title=title)
    request.session["sentences"] = s.sentences
    request.session["newStory"] = False
    return redirect('/write')


def deleteStory(request, title):
    try:
        s = Story.objects.get(title=title)
        if s.author.email == request.user.email:
            s.delete()
            if title == request.session["title"]:
                newStory(request)
            return redirect('/')
        else:
            return render(request, 'webapp/error.html',
                          context={'message': "Sorry, you don't have permission to access that story. Try logging in."})
    except:
        return render(request, 'webapp/error.html', context={'message': "Story not found."})


def write(request):
    if "prompt" not in request.session.keys():
        request.session["prompt"] = generatePrompt()

    if "editing" not in request.session.keys():
        request.session["editing"] = False

    if "sentences" not in request.session.keys():
        request.session["sentences"] = ""

    if "newStory" not in request.session.keys():
        request.session["newStory"] = True
    # title of the story in records
    if "title" not in request.session.keys():
        request.session["title"] = ""

    global sess, model, word_to_id, id_to_word

    if not model:
        sess, model, word_to_id, id_to_word = load_model(save=False)

    suggestion = ""
    sentences = request.session.get("sentences")
    editing = request.session.get("editing")
    title = request.session.get("title")
    newStory = request.session.get("newStory")

    if request.POST:
        print(request.POST)
        if request.POST.get("update"):
            print("UPDATE WOOOOOOOOOTTTTTT")
        if request.POST.get("text"):
            newSentence = request.POST["text"]
            sentences += (newSentence + "\n")

            if not editing:
                suggestion = generateSuggestion(newSentence, develop=False)

            request.session["editing"] = not editing
            request.session["sentences"] = sentences

        if request.POST.get("title"):
            title = request.POST["title"]
            request.session["title"] = title

        # STILL WORKING THIS
        # TODO: make Save button read "Saved" after saving, revert after edit
        if request.POST.get("save"):
            user = getOrCreateUser(request)

            title = request.POST["title"]
            if request.session.get("newStory"):
                print("making new Story")
                s = Story.objects.create(sentences=sentences, title=title)
                s.author = user
                s.save()
                user.stories.add(s)
                user.save()
                request.session["newStory"] = False
            else:
                s = Story.objects.get(title=request.session["title"])
                s.sentences = sentences
                s.title = title
                s.author = user
                s.save()
                user.stories.add(s)
                user.save()
            request.session["title"] = title

    elif request.GET.get("new"):
        return redirect('/new_story')

    last = ""
    if sentences != "":
        last = sentences.split("\n")[-1]

    return render(request, 'webapp/write.html',
                  context={"prompt": request.session.get("prompt"),
                  "sentences": request.session["sentences"].split("\n")[:-1],
                  "suggestion": suggestion, "last":last,
                  "title":request.session["title"]})


def about(request):
    return render(request, 'webapp/about.html')


def team(request):
    return render(request, 'webapp/team.html')


def error(request, message):
    return render(request, 'webapp/error.html', context={'message': message})


def saves(request):
    # need to implement the title and author thing....
    # author needs users/auth
    stories = Story.objects.all()
    #stories = [s.sentences for s in stories]
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


def generateSuggestion(newSentence, develop=False):
    if develop:
        return "look! a {} {}".format(random.choice(ADJECTIVES), random.choice(ANIMALS))
    try:
        suggestion = generate_text(sess, model, word_to_id, id_to_word, seed=newSentence)
    except Exception as e:
        print("ERROR (suggestion generation)")
        suggestion = e
    return suggestion

