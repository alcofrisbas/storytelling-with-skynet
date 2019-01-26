from django.contrib import admin
from webapp.models import Story
from webapp.models import User
from django.contrib.auth.admin import UserAdmin
# Register your models here.

admin.site.register(Story)
admin.site.register(User, UserAdmin)
