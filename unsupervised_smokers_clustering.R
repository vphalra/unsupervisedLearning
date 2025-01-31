library(dplyr)
library(ggplot2)

df <- read.csv("smoking.csv")
head(df)
dim(df)
summary(df)

#cleaning data
df$amt_weekends[is.na(df$amt_weekends)] <- 0
df$amt_weekdays[is.na(df$amt_weekdays)] <- 0

head(df,10)

#normalization
cnorm <- c("age","amt_weekdays","amt_weekends")
nc <- scale(df[,cnorm])
ndf <- data.frame(nc)

#HC model
dist_matrix <- dist(nc)
hcmodel <- hclust(dist_matrix, method="ward.D2")

#dendogram
plot(hcmodel, labels=FALSE, hang=-1, main="Dendrogram For HC Model")

#clusters
c <- 7
clusters <- cutree(hcmodel, k = c)

df$Cluster <- clusters
ndf$Cluster <- clusters
head(ndf)

#summary statistics
clusterlist <- split(ndf,ndf$Cluster)
clusterlist

clusterstats <- list()
for(i in 1:7) {
  clusterdata <- clusterlist[[as.character(i)]]
  clustersummary <- summary(clusterdata)
  clusterstats[[i]] <- clustersummary
}
for (i in 1:7){
  cat("\nSummary for Cluster", i, "\n")
  print(clusterstats[[i]])
}

#visualization
ggplot(df, aes(x = as.factor(Cluster))) +
  geom_bar(fill = "orange") +
  labs(title = "Bar Plot of Cluster Counts", x = "Cluster", y = "Count")

ggplot(df, aes(x = as.factor(Cluster), y = age, color = as.factor(Cluster))) +
  geom_boxplot() +
  labs(title = "Box Plot of Age by Cluster", x = "Cluster", y = "Age")

ggplot(df, aes(x = age, y = amt_weekdays, color = as.factor(Cluster))) +
  geom_point() +
  labs(title = "Scatter Plot of Age and # of Cigs on Weekdays", x = "Age", y = "Amount on Weekdays")

ggplot(df, aes(x = age, y = amt_weekends, color = as.factor(Cluster))) +
  geom_point() +
  labs(title = "Scatter Plot of Age and # of Cigs on Weekends", x = "Age", y = "Amount on Weekends")


#randomsmoker
set.seed(420)
rsmoker <- sample(df$X,size = 1)
rsmoker

emma <- df%>%
  filter(X==773)
emma
ggplot(df[df$Cluster == 6, ], aes(x = marital_status)) +
  geom_bar(fill = "red") +
  theme_minimal() +
  labs(title = "Distribution of Marital Status in Cluster 6")

#model 2
data <- read.csv("smoking.csv")
data$amt_weekends[is.na(data$amt_weekends)] <- 0
data$amt_weekdays[is.na(data$amt_weekdays)] <- 0


#data prep
data <- data%>%
  mutate(CigsPerWeek = amt_weekdays+amt_weekends)
kc <- c("CigsPerWeek", "age")
normv <- scale(data[,kc])
dfv <- data.frame(normv)
dfv

#hc model
dist <- dist(normv)
model2 <- hclust(dist, method = "ward.D2")

#dendogram
plot(model2, labels=FALSE,hang=-1,main="Dendogram for 2nd HC Model")

#clusters 
clster <- cutree(model2, k = c)
dfv$Cluster <- clster
data$Cluster <- clster

#summary statistics
clsterl <- split(dfv,dfv$Cluster)

clsterstats <- list()
for(i in 1:7) {
  clsterdata <- clsterl[[as.character(i)]]
  clstersummary <- summary(clsterdata)
  clsterstats[[i]] <- clstersummary
}
for (i in 1:7){
  cat("\nSummary for Cluster", i, "\n")
  print(clsterstats[[i]])
}

#visualizations
ggplot(data, aes(x = as.factor(Cluster))) +
  geom_bar(fill = "orange") +
  labs(title = "Bar Plot of Cluster Counts", x = "Cluster", y = "Count")

ggplot(data, aes(x = as.factor(Cluster), y = age, color = as.factor(Cluster))) +
  geom_boxplot() +
  labs(title = "Box Plot of Age by Cluster", x = "Cluster", y = "Age")

ggplot(data, aes(x = age, y = CigsPerWeek, color = as.factor(Cluster))) +
  geom_point() +
  labs(title = "Scatter Plot of Age and Cigarettes per Week", x = "Age", y = "Cigarettes per Week")



