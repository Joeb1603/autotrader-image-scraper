#pip install requestions
#pip install bs4
#pip install selenium

from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import webbrowser
from PIL import Image
import os 
import time
import shutil

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, StaleElementReferenceException

options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_experimental_option('excludeSwitches', ['enable-logging'])
options.page_load_strategy = 'eager'
driver = webdriver.Chrome(options=options)

driver.implicitly_wait(0.5)

print("\n")

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

    try:
        driver.find_element(By.ID, 'image-0').find_element(By.TAG_NAME, 'button').click()
    except NoSuchElementException:
        print("show next image button not showing on first image (fine if only 1 image there)\n")

    from pathlib import Path
    Path("images").mkdir(parents=False, exist_ok=True)

    images = driver.find_element(By.CLASS_NAME, "fullscreen-modal").find_elements(By.CLASS_NAME, "slick-slide")
    
    timed_out_already = False

    while True:
        try:
            driver.find_element(By.CLASS_NAME, "show-gallery").find_element(By.XPATH, '//button[@aria-label="Next image"]').click()
            #driver.find_element(By.CLASS_NAME, "show-gallery").find_element(By.CLASS_NAME, "iVKNPy").click()
        except ElementNotInteractableException:
            break
        except NoSuchElementException:
            print("This should probably only happen if theres only one image\n")
            break


    img_elements = driver.find_element(By.CLASS_NAME, "fullscreen-modal").find_elements(By.TAG_NAME, "img")

    if (len(images)!=len(img_elements)):
        print("Not all the images are loaded, something has gone wrong\n")

    current_index=len(os.listdir("images"))
    for element in img_elements:
        current_url = element.get_attribute("src")

        try:
            data = requests.get(current_url).content 
        except requests.exceptions.ConnectionError:
            if timed_out_already:
                break
            print("Too many requests, taking a break")
            time.sleep(60*5)
            timed_out_already = True
        
        with open(f'images/{current_index}.png','wb') as handler:
            handler.write(data) 


        current_index+=1

def process_page(url):
    load_page(url)
    #print_page_details()
    if cookie_box_visible():
        accept_cookies()
    try:
        collect_page_pics()
    except NoSuchElementException: # StaleElementReferenceException
        print("Something went wrong collecting images from the page, skipping")
    except StaleElementReferenceException as e:
        print("Not really sure what causes this, but it's bad - skipping to next page")
        print(e)
        print("\nError printed above\n\n")


def process_list(url_list):
    for current_url in url_list:
        process_page(current_url)

def get_all_links():
    link_elements = driver.find_element(By.XPATH, "//ul[@data-testid='desktop-search']").find_elements(By.XPATH, '//a[@data-testid="search-listing-title"]')
    page_urls = [element.get_attribute("href") for element in link_elements]
    return page_urls

def process_results_page(input_arg_dict, num_pages):
    arguments = ['='.join([x,y]) for x,y in input_arg_dict.items()]
    arguments_combined = '&'.join(arguments)

    for page_num_idx in range(num_pages):
        current_page = page_num_idx+1
        results_page = f'https://www.autotrader.co.uk/car-search?postcode=NE16an&{arguments_combined}&page={current_page}'
        
        load_page(results_page)

        if cookie_box_visible():
            accept_cookies()

        max_pages = int(driver.find_element(By.XPATH, '//p[@data-testid="pagination-show"]').text.split(" ")[-1])
        
        url_list = get_all_links()
        process_list(url_list)

        if current_page>=max_pages:
            print("max number of pages reached\n")
            break

def delete_old_dir():
    try:
        shutil.rmtree("images")
    except OSError:
        print("error occured deleting folder")

def run_multipage():
    argument_dict = {"make":"Mazda",
                    "model":"MX-5"}
    pages_to_collect = 100

    start_time = time.time()

    process_results_page(argument_dict, pages_to_collect)

    time_taken = int(time.time() - start_time)
    print(f"\nDone!\nCollected {len(os.listdir('images'))} images in {int(time_taken/3600)} hours, {int(time_taken/60)} minutes, {int(time_taken%60)} seconds")
    # Runs at roughly 1 page every 2 minutes

delete_old_dir()
run_multipage()
