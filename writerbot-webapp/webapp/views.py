from django.shortcuts import render
from django.http import HttpResponse
from webapp.models import Story
from RNN.test import load_model, generate_text
import random

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
        sess, model, word_to_id, id_to_word = load_model()

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


def generatePrompt(curPrompt=""):
    topics = ["a hotheaded penguin", "a wizened chihuahua", "a murderous toucan",
              "an exponentially multiplying swarm of beagles"]
    curTopic = curPrompt
    while curTopic == curPrompt:
        curTopic = "Write about " + topics[random.randrange(0, len(topics))]
    return curTopic


def generateSuggestion(newSentence):
    suggestion = generate_text(sess, model, word_to_id, id_to_word, newSentence)
    return suggestion
