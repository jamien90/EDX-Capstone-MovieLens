library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(tidyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Dimension of edx
dim(edx)

# Summary of edx
summary(edx)

# Movies rating summary
edx %>% group_by(rating) %>% count()

# Number of movies
n_distinct(edx$movieId)

# Number of users
n_distinct(edx$userId)

# Number for different genres
genres = c("Drama", "Comedy", "Thriller", "Romance")
sapply(genres, function(g) {
  sum(str_detect(edx$genres, g))
})

# Movie ranking in rating
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

# Distribution of rating
edx %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))

##### Model 1 - Average Edx rating ####

# Mean rating of dataset
mu <- mean(edx$rating)
mu

# RMSE Function
RMSE <- function(validation, y_hat){
  sqrt(mean((validation - y_hat)^2))
}

# RMSE_1 testing
rmse_1 <- RMSE(validation$rating, mu)
rmse_1

rmse_results <- data_frame(method = 'Model 1 - Average rating', RMSE = rmse_1)

# Results compilation
rmse_results %>% knitr::kable()

##### Model 2 - Movie Effect #####

# Movie effect is taken into account  for this model
# b_i is the mean difference between movie rating and average rating
avg_movies <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# RMSE_2 testing
pred_ratings <- mu +  validation %>%
  left_join(avg_movies, by='movieId') %>%
  pull(b_i)
rmse_2 <- RMSE(pred_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method='Model 2 - Movie Effect',  
                                     RMSE = rmse_2 ))

# Results compilation
rmse_results %>% knitr::kable()

##### Model 3 - Movie + User Effect #####

# Movie and user effects are taken into account  for this model
# Plot User Effect #
avg_user <- edx %>% 
  left_join(avg_movies, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))
avg_user %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))


avg_user <- edx %>%
  left_join(avg_movies, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# RMSE_3 testing
pred_ratings <- validation%>%
  left_join(avg_movies, by='movieId') %>%
  left_join(avg_user, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_3 <- RMSE(pred_ratings, validation$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 3 - Movie + User Effect",  
                                     RMSE = rmse_3))

# Results compilation
rmse_results %>% knitr::kable()

##### Model 4 - Regularization of Movie + User Effect #####

# Movie and user effects are regularised in this model
# Fine tuning of lambda
lambdas <- seq(0, 10, 0.25)

tuning <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  pred_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(pred_ratings, validation$rating))
})


# Plot of lambda tuning and find optimal lambda                                                   
qplot(lambdas, tuning, main="Values of RMSE vs. parameter Lambda",
      xlab="Lambda", ylab="RMSE")  

opt_lambda <- lambdas[which.min(tuning)]
opt_lambda

# RMSE_4 testing                                                        
rmse_4 <- min(tuning)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Model 4 - Regularization of Movie + User Effect",  
                                     RMSE = rmse_4))

##### Final compilation of results #####                                                     
rmse_results %>% knitr::kable()
