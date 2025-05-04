import pandas as pd
from google_play_scraper import reviews, Sort

scrapreviews, continue_token = reviews(
    'com.amazon.mShop.android.shopping',
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=10000,
)

result, _ = reviews(
    'com.amazon.mShop.android.shopping',
    continuation_token=continue_token,
)

reviews_data = pd.DataFrame(scrapreviews)

reviews_data.to_csv('amazon_reviews.csv', index=False)