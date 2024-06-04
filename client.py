import requests

url = "http://127.0.0.1:8000/ask"
payload = {
    "question": input("Ask your question?  : ")
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())