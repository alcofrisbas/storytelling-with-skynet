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

#TODO: when user logs in, redirect to the page they logged in from
#TODO: figure out how to clear empty stories and expired session data

# Create your views here.
def home(request):
    if request.user.is_authenticated:
        user = getOrCreateUser(request)
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

    s = Story.objects.create(sentences="", title="")
    if request.user.is_authenticated:
        user = getOrCreateUser(request)
        s.author = user
        s.save()
        user.stories.add(s)
        user.save()
    request.session["editing"] = False
    request.session["prompt"] = generatePrompt(request.session.get("prompt"))
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
        print("starting new story")
        newStory(request)

    if "developer" not in request.session.keys():
        request.session["developer"] = False

    global sess, model, word_to_id, id_to_word

    # I was tired of loading TODO: UNCOMMENT ME
    # if not model:
    #     sess, model, word_to_id, id_to_word = load_model(save=False)

    story = Story.objects.get(id = request.session["story_id"])
    suggestion = ""
    editing = request.session["editing"]

    if request.POST:
        print("======== ===== ===== ====")
        print(request.POST.keys())
        if request.POST.get("text"):
            newSentence = request.POST["text"]
            story.sentences += newSentence.strip()+ "\n"
            story.save()

            if not editing:
                suggestion = generateSuggestion(newSentence, develop=request.session["developer"])

            request.session["editing"] = not editing

        if request.POST.get("title"):
            story.title = request.POST["title"]

        # TODO: make Save button grayed out after saving, revert after edit
        if request.POST.get("save"):
            if request.user.is_authenticated:
                user = getOrCreateUser(request)
                title = request.POST["title"]
                s = Story.objects.get(id = request.session["story_id"])
                s.title = title
                s.save()
            else:
                return render(request, 'webapp/error.html', context={'message': "Please log in before trying to save a story."})

        # same functionality as "Start a new story button"
        if request.POST.get("new"):
            return redirect('/new_story')

        if request.POST.get("side-open"):
            print("open story Pressed")

        if request.POST.get("side-settings"):
            print("settings story Pressed")

        if request.POST.get("side-toggle"):
            print("toggle story Pressed")
            request.session["developer"] = not request.session["developer"]
            print("dev mode", request.session["developer"])

        if request.POST.get("sentence-content"):
            print("-- -- -- -- --")
            print(request.POST["sentence-content"])
            print("-- -- -- -- --")
            story.sentences = request.POST["sentence-content"].strip()
            story.save()

    elif request.GET.get("new"):
        return redirect('/new_story')

    last = ""
    if story.sentences != "":
        last = story.sentences.split("\n")[-1]

    power = "glow"
    if request.session["developer"]:
        power = ""

    return render(request, 'webapp/write.html',
                  context={"prompt": request.session["prompt"],
                  "sentences": [s.strip() for s in story.sentences.split("\n")[:-1]],
                  "suggestion": suggestion, "last":last,
                  "title": story.title, "power":power})


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


def generateSuggestion(newSentence, develop=False):
    if develop:
        return "look! a {} {}".format(random.choice(ADJECTIVES), random.choice(ANIMALS))
    try:
        suggestion = generate_text(sess, model, word_to_id, id_to_word, seed=newSentence)
    except Exception as e:
        print("ERROR (suggestion generation)")
        suggestion = e
    return suggestion
