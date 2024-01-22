#pip install requestions
#pip install bs4

from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import webbrowser

page_url = "https://www.geeksforgeeks.org/"
#"https://www.autotrader.co.uk/car-details/202310142996962?sort=relevance&body-type=Convertible&postcode=NE16an&advertising-location=at_cars&fromsra"

page_data = requests.get(page_url) 

soup = BeautifulSoup(page_data.content, "html.parser")
#print(soup)

images = soup.find_all('img')

if len(images)>0:
    first_image = images[0]['src']
    print(first_image)
    webbrowser.open(first_image)
else:
    print("No images found")