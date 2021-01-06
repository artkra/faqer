from django.http.response import JsonResponse
from django.views import View

from faqer.services.cluster.suggest import categories_service


class AskView(View):

    def get(self, request, *args, **kwargs):
        question = request.GET.get('q')
        if question:
            return JsonResponse(categories_service.predict_category(question), safe=False)
        return JsonResponse(categories_service.UNCATEGORIZED, safe=False)


class CategoriesView(View):

    def get(self, request, *args, **kwargs):
        _type = request.GET.get('type')
        if _type == 'cache':
            return JsonResponse(categories_service.cached_categories, safe=False)
        if _type == 'suggested':
            return JsonResponse(categories_service.suggested_categories, safe=False)
        return JsonResponse(categories_service.categories, safe=False)
