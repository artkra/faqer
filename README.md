Run from image:

```bash
docker run -p <YOUR_PORT>:8000 artkra/faqer:latest
```

That starts dummy HTTP server on desired port.
Question category is suggested via GET request on defined endpoint (`/ask`).
Question body is expected as `q` query parameter.

E.g.:

```bash
GET http://localhost:<YOUR_PORT>/ask/?q=<YOUR QUESTION GOES HERE>
```

Response is a json object containing category id, name, possible answer/solution and minimum distance (for debugging purposes) which is considered to be a stupid replacement for probability:

```json
{
    "id": 5,
    "category_name": "dev ops",
    "answer": "Try docs (https://confluence.iponweb.net/display/DEVOPS) or ask here: #devops-support",
    "distance": 0.3538993000984192
}
```