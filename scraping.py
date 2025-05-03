import pandas as pd
from google_play_scraper import reviews, Sort

# Scraping data reviews from Tokokpedia Android app
scrapreviews, continue_token = reviews(
    'com.amazon.mShop.android.shopping',
    lang='en',
    country='us',
    sort=Sort.NEWEST,
    count=10000,
)

# Continue scraping if there are more reviews
result, _ = reviews(
    'com.amazon.mShop.android.shopping',
    continuation_token=continue_token,
)

# Dataframe to store reviews
reviews_data = pd.DataFrame(scrapreviews)

# Save reviews to CSV
reviews_data.to_csv('amazon_reviews.csv', index=False)