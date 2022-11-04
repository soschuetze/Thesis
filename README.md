# Thesis

The intense emotions surrounding the acquittal of George Zimmerman in the murder of Trayvon Martin and the surge in popularity of Twitter in 2013 coincided to create an environment suited to a modern wave of social movements. The formation of the #BlackLivesMatter movement found its homebase on Twitter due to the platform’s accessibility, popularity, and potential for both national and global impact. This extensive impact is visible in the formation of other recent social movements such as #MeToo and #StopAsianHate, where a direct influence by the #BlackLivesMatter movement is undeniable. This significance has prompted many researchers to analyze the role of social media in new social movements. One such paper includes that by Al Tillery entitled “What Kind of Movement is Black Lives Matter? The View from Twitter” in which Tillery analyzed over 18,000 tweets from BLM organizations across the country to determine the kind of response generated through the movement, the different frames contained within the movement, and the types of advocacy promoted by the organizations. 

A significant motivator of my honors thesis is to expand upon Tillery’s work through the application of data science methods unique to working with Big Data. Given the use of social media platforms, new social movements, such as #BlackLivesMatter, provide the opportunity to process vast amounts of data. Because I am able to process millions of tweets, I can analyze all Twitter accounts ranging from “everyday” users and grassroots organizations to politicians and elites. Additionally, this thesis will cover the entire length of the #BlackLivesMatter movement from its inception in 2013 until December of 2021. 

Another important component contributing to my thesis is the work already completed by Wellesley’s CAPS lab. This lab, run by computer science professor Eni Mustafaraj and political science professor Maneesh Arora, collected over 21 million tweets that contain hashtags related to the #BlackLivesMatter movement. Once they obtained these tweets, student researchers classified subsets of the tweets with the applicable call to action contained in the body of the tweet: within the system, disruptive, spreading information, moral encouragement, etc. The lab then conducted exploratory data analysis to understand the presence of intersectional tweets, the utilization of hashtags, and how tweets have changed along with the movement. 
My thesis consists of two distinct phases - model creation and data analysis. The first phase of research currently underway focuses on understanding the idea of what constitutes new social movements and situating the Black Lives Matter movement within that context. I am completing my background research and literature review looking at how #BlackLivesMatter came to gain recognition on Twitter, how the use of the hashtag has risen and declined as more events related to the movement have occurred, and how data science has already been used with respect to this and other social movements. I have also begun exploring the data by looking at visualizations produced from unsupervised methods, such as word clouds and bar charts representing LDA models, to understand the most popular themes contained in the tweets. Finally, I am researching natural language processing classification models, which are models that take in text as data and determine which category the text best falls into. There are many existing classification models; however, tweets present a challenge given their limited length making it harder for models to grasp their context. I will train and test these researched models, including Naive Bayes, Support Vector Machines, and GPT-3, on the twitter data, and I will determine which one suits the data best through the use of measures such as precision, recall, and F-scores. This final model will then label the tweets with the appropriate call to action.

The second phase of research will use the labeled data for a broader data analysis regarding the #BlackLivesMatter movement. There will be three main research questions relating to the data analysis. The first question focuses on how the forms of advocacy vary with the type of Twitter user. Do individuals want something different than elites or grassroots activist groups, and does a person’s gender change the way they tweet about the movement? The next question is derived from Tillery’s paper. Because of the claims made by some news outlets and organizations saying the #BlackLivesMatter movement promotes violence, Tillery studied whether there was any evidence of these statements, and his analysis saw no tweets that supported these claims. My research aims to determine if this finding is still supported when taking into account more years and volume of tweets. And if this finding does not hold up, I am interested to see under what conditions that is the case. Lastly, there has been a recent feeling of frustration among supporters of the movement due to the lack of any tangible systemic change. The final research question therefore pertains to determining if there is any evidence of this sentiment in the Twitter discourse. All of these research questions will involve analysis of time series data which will allow me to make conclusions about how any of these findings have changed over the course of the movement. 
