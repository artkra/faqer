import json

from django.http.response import JsonResponse
from django.views import View

from faqer.services.cluster.suggest import categories_service


class AskView(View):

    def get(self, request, *args, **kwargs):
        question = request.GET.get('q')
        if question:
            response_data = categories_service.predict_category(question)
            if 'keywords' in response_data:
                del response_data['keywords']
            return JsonResponse(response_data, safe=False)
        return JsonResponse(categories_service.UNCATEGORIZED, safe=False)


class CategoriesView(View):

    def get(self, request, *args, **kwargs):
        _type = request.GET.get('type')
        if _type == 'cache':
            return JsonResponse(categories_service.cached_categories, safe=False)
        if _type == 'suggested':
            return JsonResponse(categories_service.suggested_categories, safe=False)
        return JsonResponse(categories_service.categories, safe=False)

    def put(self, request, *args, **kwargs):
        categories_list = json.loads(request.body)
        if not isinstance(categories_list, list):
            return JsonResponse(
                {
                    'error': 'Request must contain a list of category objects ("id", "category_name", "keywords", "response")'
                },
                status=400
            )
        categories_service.update_categories(categories_list)
        return JsonResponse({}, status=200)
