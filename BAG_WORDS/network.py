import requests
ip = "192.168.119.167" #ip gerado pelo ESP32
def send_request(string):
    url = f'http://{ip}/{string}'
    response = requests.get(url)
    print(response)


send_request("esque")