# Project-Fletcher: Creating recommender systems from Yelp reviews

Originally, the focus of this project was to use unsupervised learning and NLP to create a recommender system based on Yelp reviews.  However, due to the nature of the scraping function, the data that was mined from Yelp was insufficient to create a collaborative filter recommender system.  As a result, the mined data was used to create a content-based recommender system instead.  This system takes in a user input and outputs the top matching businesses (cafes in this case) and the corresponding reviews.  

Since a collaborative filter recommender system was still an objective that I wanted to complete with this project, I decided to supplement my mined data with the publicly available data from Yelp.  With Yelp's data, the collaborative filter recommender system was created.

The scripts to the recommender systems can be found in the "modeling" folder while the scraping functions can be found in the "scraping" folder.
