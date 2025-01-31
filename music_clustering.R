#extracting circles and place into dataframe
spot <- read.csv("spot23.csv")

#read spotify, factor target, and remove na
spotify <- read.csv("spotify.csv")
spotify$target <- as.factor(spotify$target)
str(spotify)

#check for na val
naval <- sapply(spotify, function(x) sum(is.na(x)))
print(naval)

#correct names
names(spot)[names(spot) == "danceability_."] <- "danceability"
names(spot)[names(spot) == "energy_."] <- "energy"
names(spot)[names(spot) == "speechiness_."] <- "speechiness"
names(spot)[names(spot) == "valence_."] <- "valence"
names(spot)[names(spot) == "acousticness_."] <- "acousticness"
names(spot)[names(spot) == "liveness_."] <- "liveness"

#convert values to decimals 
spot$danceability <- spot$danceability / 100
spot$energy <- spot$energy / 100
spot$speechiness <- spot$speechiness / 100
spot$valence <- spot$valence / 100
spot$acousticness <- spot$acousticness / 100
spot$liveness <- spot$liveness / 100


#df of circles with revised values and name
circles <- spot[grepl("Circles", spot$track_name, ignore.case = TRUE),]
circles

#data partitioning of spotify dataset
set.seed(420) 
indices <- sample(1:nrow(spotify), size = 0.6 * nrow(spotify))
training_set <- spotify[indices, ]
validation_set <- spotify[-indices, ]

#t-test
ttest <- list()
variables <- c("danceability", "energy", "speechiness", "valence", "acousticness", "liveness")

for (var in variables) {
  formula <- as.formula(paste(var, "~ target"))
  ttest[[var]] <- t.test(formula, data = training_set)
}
ttest

#removing energy, liveness, trackname, and artist
training_set <- training_set[, !(names(training_set) %in% c("energy", "liveness"))]

#prepping circles and training
circles_features <- circles[c("danceability", "valence", "acousticness", "speechiness")]
training_features <- training_set[c("danceability", "valence", "acousticness", "speechiness")]

library(caret)
preProcValues <- preProcess(training_features[, c("danceability", "valence", "acousticness", "speechiness")], method = c("center", "scale"))
normtrain <- predict(preProcValues, training_features)
normcircle <- predict(preProcValues, circles_features)

#k-nn model
library(FNN)
tlabel <- training_set$target
trainf <- normtrain[, !(names(normtrain) %in% c("target"))]
prediction <- knn(train = trainf, test = normcircle, cl = tlabel, k = 7)
print(prediction)

#nearest neighbor songs
nearestsongs <- spotify[c(1077, 238, 257, 743, 334, 145, 427), c("song_title", "artist", "target")]
print(nearestsongs)

vfeature <- validation_set[c("danceability", "valence", "acousticness", "speechiness")]
vpreProcValues <- preProcess(vfeature[, c("danceability", "valence", "acousticness", "speechiness")], method = c("center", "scale"))
normv <- predict(vpreProcValues, vfeature)
vlabel <- validation_set$target

accuracies <- sapply(1:40, function(k) {
  predicted_labels <- knn(train = trainf, test = normv, cl = tlabel, k = k)
  mean(predicted_labels == vlabel)
})

# Find optimal k with the highest accuracy
optk <- which.max(accuracies)
optk

# Plotting accuracies vs k-values
plot(1:40, accuracies, 
     col = 'blue', 
     xlab = "k-value", 
     ylab = "Accuracy", 
     main = "Optimal K-Value",
     pch = 1,  # This sets the point shape to solid circles
     cex = 1) # This sets the size of the points

#optimal k k-nn model
optp <- knn(train = trainf, test = normcircle, cl = tlabel, k = 24)
print(optp)
nearestoptsongs <- spotify[c(1077, 238, 257, 743, 334, 145, 427, 815, 651, 1170, 1051, 974, 1103, 279, 42, 755, 996, 534, 995, 1123, 808, 117, 98, 1071), c("song_title", "artist", "target")]
print(nearestoptsongs)
