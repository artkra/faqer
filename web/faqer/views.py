from django.http.response import JsonResponse
from django.views import View


class AskView(View):

    def post(self, request, *args, **kwargs):
        return JsonResponse(data={})


class CategoriesView(View):

    def get(self, request, *args, **kwargs):
        return JsonResponse({})
