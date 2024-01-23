#pip install requestions
#pip install bs4
#pip install selenium

from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import webbrowser
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=options)

print("\n\n\n")

# Set stuff up
mx5_1 = "https://www.autotrader.co.uk/car-details/202310142996962"
mx5_2 = "https://www.autotrader.co.uk/car-details/202401125480291"

driver.get(mx5_1)

title = driver.title
print(f"The title is: {title}")

driver.implicitly_wait(0.5)

# Accept the cookies
cookie_frame_id = "sp_message_iframe_982392" 
cookie_button_id = "/html/body/div/div[2]/div[4]/button[3]"


cookie_frame = driver.find_element(By.ID, cookie_frame_id)

wait = WebDriverWait(driver, 1)
wait.until(EC.frame_to_be_available_and_switch_to_it(cookie_frame))

cookie_button = wait.until(EC.element_to_be_clickable((By.XPATH, cookie_button_id)))
cookie_button.click()


driver.implicitly_wait(0.5)

# Click onto the firest image of the image gallery
buttons_to_click = ['/html/body/div[1]/main/div/div[2]/article/section[1]/div/div/div[1]/button[2]',
                        "/html/body/div[2]/div[2]/div/section/div[2]/div/div/div/div[2]/ul/li[1]/button/div/div/div/div/div/picture/div/img"]


for button in buttons_to_click:
    driver.find_element(By.XPATH, button).click()

driver.implicitly_wait(0.5)

# Go and click through all the images and download them 
images_parent_div_id = '/html/body/div[3]/div[2]/div/section/div[2]/div/div/div/div[2]/div/div/div'
next_img_button_id = "/html/body/div[3]/div[2]/div/section/div[2]/div/div/div/div[2]/div/button[2]"

from pathlib import Path
Path("images").mkdir(parents=False, exist_ok=True)

x=0
while True:
    x+=1
    try:
        current_url = driver.find_element(By.XPATH, images_parent_div_id).find_element(By.CLASS_NAME, "slick-current").find_element(By.TAG_NAME, 'img').get_attribute("src")
        
        data = requests.get(current_url).content 
        with open(f'images/{x}.png','wb') as handler:
            handler.write(data) 

        driver.find_element(By.XPATH, next_img_button_id).click()
        driver.implicitly_wait(0.5)
    except:
        print("All images should be done")
        break

print("\n\n\n")