# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 00:33:11 2024

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
 

options = Options() 

options.add_argument('--headless')
options.add_argument('--window-size=1920x1080')
options.add_argument('--disable-gpu') 

# driver 객체 생성 
driver = wd.Chrome(service=Service(ChromeDriverManager().install()))
dir(driver)

# 크롤링할 리뷰 사이트 URL
url = 'https://www.amazon.com/Paris-Makeup-True-Natural-Glow-Illuminator-Highlighter-Day-Radiant-Glow/product-reviews/B074PTZCNX/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'  # 리뷰 url 
#url = 'https://www.naver.com'
driver.get(url)
driver.implicitly_wait(10) 
res = driver.page_source 
obj = bs(res, 'html.parser')


# (1) 리뷰 갯수 
rev_cnts = obj.select('div[data-hook="cr-filter-info-review-rating-count"]')

rev_cnts = rev_cnts[0].get_text(strip=True)
rev_cnts = rev_cnts.replace(',', '')
rev_cnts = re.findall('\d+', rev_cnts)
rev_cnts = int(rev_cnts[1])
rev_cnts

page_number = 1
titles = [] 
stars = [] 
dates = [] 
contents = [] 

'''
source = driver.page_source 
bs_obj = bs(source, 'html.parser')
date = bs_obj.findAll('span', {'data-hook':'review-date'})
print(date)
    
'''

while len(titles) < 100:  #  3 대신 rev_cnts
    time.sleep(3) 
    source = driver.page_source 
    bs_obj = bs(source, 'html.parser')
    
    # 리뷰 타이틀 가져오기 
    for i in bs_obj.findAll('a', {'data-hook':'review-title'}):
        titles.append(i.get_text().strip().split('\n')[1])

    # 리뷰 날짜 가져오기 
    for d in bs_obj.findAll('span', {'data-hook': 'review-date'}):
        # 리뷰 날짜 문자열 가져오기
        date_str = ''.join(d.get_text().split(' ')[-3:])  # "August27,2024" 형식
        # "August27,2024" -> "August 27 2024"로 변환
        # 쉼표 제거
        date_str1 = date_str.split(',')
        month_date = date_str1[0]
        year = date_str1[1].strip() 
        # 숫자와 문자를 분리하여 공백 추가
        month = ''.join(filter(str.isalpha, month_date))  # "August"
        day = ''.join(filter(str.isdigit, month_date)) # 1
        
        date_str = f"{month} {day} {year}" 
         
        # 문자열을 datetime 객체로 변환
        date = datetime.strptime(date_str, '%B %d %Y').date()  # 날짜 형식에 맞춰 변환
        dates.append(date)   
    
    
    # 리뷰 내용 가져오기 
    for a in bs_obj.findAll('span', {'data-hook': 'review-body'}):
        contents.append(a.get_text().strip())
    
    # 리뷰 평점 가져오기 
    for s in bs_obj.findAll('i', {'data-hook': 'review-star-rating'}):
        stars.append(s.get_text()[0])
        
    for s in bs_obj.findAll('i', {'data-hook': 'cmps-review-star-rating'}):
        stars.append(s.get_text()[0])    
        
    if len(titles) == 100:
        
        break
    
    # 다음 페이지로 넘어가는 버튼 찾기
    try:
        page_number += 1  # 페이지 번호 증가
        next_button_xpath = f'//a[contains(@href, "pageNumber={page_number}")]'
        
        # 다음 버튼 클릭
        next_button = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, next_button_xpath))
        )
        next_button.click()
    
    except Exception as e:
        print(f"다음 페이지로 넘어가는 데 실패했습니다: {e}")
        break
    
    
    
driver.close()
driver.quit() 


df = pd.DataFrame({'date':dates, 'rating':stars, 'title':titles, 'contents':contents})

df.to_csv('cosmetic_amazon_review.csv', index = False)