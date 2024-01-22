#pip install requestions
#pip install bs4
#pip install selenium

from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import webbrowser


from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.chrome.options import Options

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=options)

print("\n\n\n")
driver.get("https://www.selenium.dev/selenium/web/web-form.html")

title = driver.title
print(f"The title is: {title}")

driver.implicitly_wait(0.5)

text_box = driver.find_element(by=By.NAME, value="my-text")
submit_button = driver.find_element(by=By.CSS_SELECTOR, value="button")

text_box.send_keys("Selenium")
submit_button.click()

message = driver.find_element(by=By.ID, value="message")
text = message.text

print(f"THE TEXT IS: {text}")

print("\n\n\n")


#driver.quit()
'''driver = webdriver.Chrome()

#driver.get("https://www.autotrader.co.uk/car-details/202310142996962")
driver.get("https://www.selenium.dev/selenium/web/web-form.html")

driver.implicitly_wait(0.5)
#cookies_button_class = "message-component message-button no-children focusable sp_choice_type_11 last-focusable-el"
#cookies_button = driver.find_element(by=By.CSS_SELECTOR, value="button")

#driver.find_element(By.CLASS_NAME, "message-component message-button no-children focusable sp_choice_type_11 last-focusable-el")
#cookies_button.click()


driver.implicitly_wait(100)
driver.quit()'''

#this mad

'''page_url = "https://www.geeksforgeeks.org/"
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
    print("No images found")'''

