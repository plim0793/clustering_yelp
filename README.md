# Creating recommender systems from Yelp reviews and ratings

## Content-Based Recommender System

Originally, the focus of this project was to use unsupervised learning and NLP to create a recommender system based on Yelp reviews.  The mined data was used to create a content-based recommender system.  This system takes in a user input and outputs the top matching businesses (cafes in this case) and the corresponding reviews. 

## Collaborative-Filter Recommender System

A collaborative filter recommender system was still an objective that I wanted to complete with this project, I decided to supplement my mined data with the publicly available data from Yelp.  With Yelp's data, the collaborative filter recommender system was created.

The scripts to the recommender systems can be found in the "modeling" folder while the scraping functions can be found in the "scraping" folder.  The Flask apps for the two recommender systems can be found in the "collab_app" and "content_app" folders.
