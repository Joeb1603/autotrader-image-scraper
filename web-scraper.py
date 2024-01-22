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
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=options)

print("\n\n\n")

mx5_1 = "https://www.autotrader.co.uk/car-details/202310142996962"
mx5_2 = "https://www.autotrader.co.uk/car-details/202401125480291"

driver.get(mx5_1)

title = driver.title
print(f"The title is: {title}")

driver.implicitly_wait(0.5)


cookie_frame_id = "sp_message_iframe_982392" #id
cookie_button_id = "/html/body/div/div[2]/div[4]/button[3]"

cookie_frame = driver.find_element(By.ID, cookie_frame_id)

wait = WebDriverWait(driver, 1)
wait.until(EC.frame_to_be_available_and_switch_to_it(cookie_frame))

cookie_button = wait.until(EC.element_to_be_clickable((By.XPATH, cookie_button_id)))
cookie_button.click()


buttons_to_click = ['/html/body/div[1]/main/div/div[2]/article/section[1]/div/div/div[1]/button[2]',
                    "/html/body/div[2]/div[2]/div/section/div[2]/div/div/div/div[2]/ul/li[1]/button/div/div/div/div/div/picture/div/img"]


for button in buttons_to_click:
    driver.find_element(By.XPATH, button).click()


next_img_button_id = "/html/body/div[3]/div[2]/div/section/div[2]/div/div/div/div[2]/div/button[2]"
while True:
    try:
        driver.find_element(By.XPATH, next_img_button_id).click()
    except:
        print("All images seen")
        break


