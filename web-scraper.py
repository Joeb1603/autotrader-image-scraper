#pip install requestions
#pip install bs4
#pip install selenium

from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import webbrowser
from PIL import Image
import os 

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=options)

driver.implicitly_wait(0.5)

print("\n\n\n")

def load_page(url):
    # Set stuff up
    driver.get(url)

def print_page_details():
    current_url = driver.current_url
    print(f"URL: {current_url}")

    title = driver.title
    print(f"The title is: {title}")


def accept_cookies():
    # Accept the cookies
    cookie_frame_id = "sp_message_iframe_982392" 
    cookie_button_id = "/html/body/div/div[2]/div[4]/button[3]"

    cookie_frame = driver.find_element(By.ID, cookie_frame_id)

    driver.switch_to.frame(cookie_frame)
    driver.find_element(By.XPATH, cookie_button_id).click()
    driver.switch_to.default_content()

def cookie_box_visible():

    try:
        cookie_frame_id = "sp_message_iframe_982392" 
        driver.find_element(By.ID, cookie_frame_id)
        return True
    except NoSuchElementException:
        return False
    
def collect_page_pics():
    
    driver.find_element(By.CLASS_NAME, 'image-gallery-button').click()

    driver.find_element(By.ID, 'image-0').find_element(By.TAG_NAME, 'button').click()

    from pathlib import Path
    Path("images").mkdir(parents=False, exist_ok=True)

    images = driver.find_element(By.CLASS_NAME, "fullscreen-modal").find_elements(By.CLASS_NAME, "slick-slide")
    
    print(f'\n\nIMAGES: {len(images)}')
    while True:
        try:
            driver.find_element(By.CLASS_NAME, "show-gallery").find_element(By.CLASS_NAME, "iVKNPy").click()
        except ElementNotInteractableException:
            print("while true loop done")
            break

    img_elements = driver.find_element(By.CLASS_NAME, "fullscreen-modal").find_elements(By.TAG_NAME, "img")
    print(f'images?:{len(img_elements)}\n\n')

    if (len(images)!=len(img_elements)):
        print("Not all the images are loaded, something has gone wrong")

    current_index=len(os.listdir("images"))
    for element in img_elements:
        current_url = element.get_attribute("src")
        data = requests.get(current_url).content 
        with open(f'images/{current_index}.png','wb') as handler:
            handler.write(data) 
        current_index+=1

def process_page(url):
    load_page(url)
    print_page_details()
    if cookie_box_visible():
        accept_cookies()
    collect_page_pics()

def process_list(url_list):
    for current_url in url_list:
        process_page(current_url)

cars = ["https://www.autotrader.co.uk/car-details/202312235033722",
        "https://www.autotrader.co.uk/car-details/202401125480291",
        "https://www.autotrader.co.uk/car-details/202310293448704",
        "https://www.autotrader.co.uk/car-details/202312154833791",
        "https://www.autotrader.co.uk/car-details/202311214162461"]

process_list(cars)

print("\n\n\n")