# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 20:17:54 2024

@author: user
"""
#  동적 웹에 유용한 셀레니움 사용
from selenium import webdriver as wd # 드라이버 
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys 
import time 
from selenium.webdriver.chrome.options import Options 
from webdriver_manager.chrome import ChromeDriverManager # 크롬드라이버 관리자  
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup as bs 
import os 
from datetime import datetime 

import requests
import re
import pandas as pd 
 
from datetime import datetime, timedelta

options = Options() 

options.add_argument('--headless')
options.add_argument('--window-size=1920x1080')
options.add_argument('--disable-gpu') 

# driver 객체 생성 
driver = wd.Chrome(service=Service(ChromeDriverManager().install()))
dir(driver)

# 크롤링할 리뷰 사이트 URL
url = 'https://www.bbc.com/news/world/europe'
driver.get(url)
driver.implicitly_wait(10) 
res = driver.page_source 
obj = bs(res, 'html.parser')

'''
# (1) 리뷰 갯수 
rev_cnts = obj.select('div[data-hook="cr-filter-info-review-rating-count"]')

rev_cnts = rev_cnts[0].get_text(strip=True)
rev_cnts = rev_cnts.replace(',', '')
rev_cnts = re.findall('\d+', rev_cnts)
rev_cnts = int(rev_cnts[1])
rev_cnts
'''
page_number = 1
titles = [] 
dates_str = [] 
dates = []
outlines = [] 

'''
source = driver.page_source 
bs_obj = bs(source, 'html.parser')
'''
###############
while len(titles) < 50:  
    time.sleep(3) 
    source = driver.page_source 
    bs_obj = bs(source, 'html.parser')
    
    alaska_section = bs_obj.find('div', {'data-testid': 'alaska-section'})
    if alaska_section:
        # 뉴스 헤드라인 가져오기 
        for i in alaska_section.findAll('h2', {'data-testid':'card-headline'}):
            titles.append(i.get_text().strip())
    
        # 뉴스 요약 가져오기 
        for o in alaska_section.findAll('p', {'data-testid': 'card-description'}):
            outlines.append(o.get_text().strip())
        
        
        # 뉴스 날짜 가져오기
        now = datetime.now()    # 현재 날짜 가져오기
        for index, d in enumerate(alaska_section.findAll('span', {'data-testid': 'card-metadata-lastupdated'})):
            if index % 2 == 0:  # 홀수 번째 원소는 짝수 인덱스에 해당
                dates_str.append(d.get_text().strip())
        
        for time_str in dates_str:
            if 'hrs ago' in time_str or 'hr ago' in time_str:
                # "X hrs ago"인 경우 시간을 추출하고 그만큼 현재 시간에서 뺌
                hours_ago = int(re.search(r'(\d+)', time_str).group(1))
                date = now - timedelta(hours=hours_ago)
            elif 'mins ago' in time_str:
                mins_ago = int(re.search(r'(\d+)', time_str).group(1)) 
                date = now - timedelta(minutes = mins_ago)
            elif 'day ago' in time_str or 'days ago' in time_str:
                # "X day ago"인 경우 날짜를 추출하고 그만큼 현재 날짜에서 뺌
                days_ago = int(re.search(r'(\d+)', time_str).group(1))
                date = now - timedelta(days=days_ago)
            else: 
                date = datetime.strptime(time_str, "%d %b %Y")
                
    
            # 결과 리스트에 추가 (YYYY-MM-DD 형식으로 변환)
            dates.append(date.strftime('%Y-%m-%d'))  
            
            
    if len(titles) == 50:
        
        break
    
    # 다음 페이지로 넘어가는 버튼 찾기
    try:
        page_number += 1  # 페이지 번호 증가
        # 클래스 이름과 페이지 번호를 기반으로 버튼 XPATH를 지정
        next_button_xpath = f'//button[contains(@class, "hrFvkk") and text()="{page_number}"]'
    
        # 다음 버튼 클릭
        next_button = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, next_button_xpath))
        )
        next_button.click()
    
    except Exception as e:
        print(f"다음 페이지로 넘어가는 데 실패했습니다: {e}")
        break

    
    
    
driver.close()
driver.quit() 



# df = pd.DataFrame({'date':dates, 'head-line':titles,  'outline':outlines})
df = pd.DataFrame({ 'head-line':titles,  'outline':outlines})



df.to_csv('bbc_europe_news.csv', index = False)