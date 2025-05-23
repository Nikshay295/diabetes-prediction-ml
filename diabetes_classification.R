
# LOAD LIBRARIES AND DATA
library(dplyr)
library(tidyverse)
library(caret)
library(e1071)
library(MASS)
library(class)
library(rpart)
library(pROC)
library(kernlab)
library(caTools)
library(corrplot)
library(GGally)
library(rpart.plot)
library(kernlab)
library(ggplot2)
library(tidyr)
library(glmnet)

# Read dataset
df <- read.csv("C:/Users/Nikshay/Desktop/diabetes.csv")
summary(df)
str(df)

# ----------------------------- Data Cleaning and Preprocessing -----------------------------
# Fix column names
names(df) <- tolower(names(df))  # Make all lowercase 

# Replace 0s with NA in appropriate columns
cols_to_fix <- c("glucose","bloodpressure","skinthickness", "insulin","bmi")
df[cols_to_fix] <- lapply(df[cols_to_fix], function(x) ifelse(x == 0, NA, x))

# Impute NA with median
df[cols_to_fix] <- lapply(df[cols_to_fix], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

df$outcome <- factor(df$outcome, levels = c(0, 1),
                     labels = c("Non-diabetic", "Diabetic"))

# ----------------------------- EDA -----------------------------
# Check class balance
# Frequency table
table(df$outcome)

# Proportions
prop.table(table(df$outcome))

# Bar plot for class distribution
ggplot(df, aes(x = outcome, fill = outcome)) +
  geom_bar() +
  labs(title = "Class Distribution", x = "Outcome", y = "Count") +
  theme_minimal()


# Histograms
df %>% 
  pivot_longer(-outcome) %>% 
  ggplot(aes(x = value, fill = outcome)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_wrap(~ name, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(title = "Histograms of Features by Diabetes Outcome")

# Correlation matrix
cor_matrix <- cor(dplyr::select(df, -outcome))
corrplot(cor_matrix, method = "color", addCoef.col = "black", tl.cex = 0.8)

# Pair plot
ggpairs(df, columns = c("glucose", "bmi", "age", "insulin", "diabetespedigreefunction"), 
        aes(color = outcome, alpha = 0.5))

# ----------------------------- PCA -----------------------------

# Scale and do PCA
pca_input <- scale(dplyr::select(df, -outcome))
pca <- prcomp(pca_input)

# Create PCA dataframe
pca_df <- as.data.frame(pca$x[, 1:2])
pca_df$outcome <- df$outcome

# PCA plot
ggplot(pca_df, aes(x = PC1, y = PC2, color = outcome)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = "PCA: First Two Principal Components") +
  theme_minimal()

# Scree Plot for PCA
scree_values <- pca$sdev^2
explained_variance <- scree_values / sum(scree_values)

plot(explained_variance, type = "b", pch = 19,
     xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     main = "Scree Plot - PCA")
abline(h = 0.1, col = "red", lty = 2)

# Use PCA features for classification
x <- as.matrix(pca$x)  # all principal components
y <- ifelse(df$outcome == "Diabetic", 1, 0)  # numeric target for glmnet

lasso_model <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(lasso_model)

# See which PCs are useful
coef(lasso_model, s = "lambda.min")


# ----------------------------- Second order interaction term Modeling -----------------------------
set.seed(123)

# Split data
index <- createDataPartition(df$outcome, p = 0.7, list = FALSE)
train <- df[index, ]
test <- df[-index, ]

# Custom summary function to include both Accuracy and ROC
custom_summary <- function(data, lev = NULL, model = NULL) {
  acc <- defaultSummary(data, lev, model)
  roc <- twoClassSummary(data, lev, model)
  c(acc, ROC = roc["ROC"])
}

# CV Control
ctrl_cv <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = custom_summary,
  savePredictions = "final"
)

# Add second-order terms
add_second_order_terms <- function(data) {
  data %>% mutate(
    glucose_sq = glucose^2,
    bmi_sq = bmi^2,
    age_sq = age^2,
    insulin_sq = insulin^2,
    glucose_bmi = glucose * bmi,
    glucose_age = glucose * age,
    bmi_age = bmi * age,
    glucose_insulin = glucose * insulin,
    bmi_insulin = bmi * insulin,
    age_insulin = age * insulin
  )
}

train2 <- add_second_order_terms(train)
test2 <- add_second_order_terms(test)

# Fix factor labels for CV compatibility
train2$outcome <- factor(train2$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
test2$outcome <- factor(test2$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))

# ----------------- LOGISTIC REGRESSION ------------------
log_model <- glm(outcome ~ ., data = train2, family = "binomial")
log_probs <- predict(log_model, newdata = test2, type = "response")
log_pred <- ifelse(log_probs > 0.5, "Diabetic", "Non-diabetic")
log_cm <- confusionMatrix(factor(log_pred, levels=c("Non-diabetic", "Diabetic")), test$outcome)
print(log_cm)
log_roc <- roc(test$outcome, log_probs)
plot(log_roc, main = "ROC Curve - Logistic Regression", col = "blue")
print(auc(log_roc))
cv_log <- train(outcome ~ ., data = train2, method = "glm", family = "binomial", metric = "ROC", trControl = ctrl_cv)
log_train_pred <- ifelse(predict(log_model, newdata = train2, type = "response") > 0.5, "Diabetic", "Non-diabetic")
log_train_acc <- mean(log_train_pred == train$outcome)

# ----------------- LDA ------------------
train_lda <- add_second_order_terms(train)
test_lda <- add_second_order_terms(test)

train_lda$outcome <- factor(train_lda$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
test_lda$outcome <- factor(test_lda$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))

lda_model <- lda(outcome ~ ., data = train_lda)
lda_pred <- predict(lda_model, newdata = test_lda)
lda_cm <- confusionMatrix(lda_pred$class, test_lda$outcome)
print(lda_cm)
lda_roc <- roc(test$outcome, as.numeric(lda_pred$posterior[,2]))
plot(lda_roc, main = "ROC Curve - LDA", col = "green")
print(auc(lda_roc))
cv_lda <- train(outcome ~ ., data = train_lda, method = "lda", metric = "ROC", trControl = ctrl_cv)
lda_train_pred <- predict(lda_model, newdata = train_lda)$class
lda_train_acc <- mean(lda_train_pred == train_lda$outcome)

# ----------------- QDA ------------------
train_qda <- add_second_order_terms(train)
test_qda <- add_second_order_terms(test)

train_qda$outcome <- factor(train_qda$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
test_qda$outcome <- factor(test_qda$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
qda_model <- qda(outcome ~ ., data = train_qda)
qda_pred <- predict(qda_model, newdata = test_qda)
qda_cm <- confusionMatrix(qda_pred$class, test_qda$outcome)
print(qda_cm)
qda_roc <- roc(test$outcome, as.numeric(qda_pred$posterior[,2]))
plot(qda_roc, main = "ROC Curve - QDA", col = "red")
print(auc(qda_roc))
cv_qda <- train(outcome ~ ., data = train_qda, method = "qda", metric = "ROC", trControl = ctrl_cv)
qda_train_pred <- predict(qda_model, newdata = train_qda)$class
qda_train_acc <- mean(qda_train_pred == train_qda$outcome)

# ----------------- NAIVE BAYES ------------------
train_nb <- add_second_order_terms(train)
test_nb <- add_second_order_terms(test)

train_nb$outcome <- factor(train_nb$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
test_nb$outcome <- factor(test_nb$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
nb_model <- naiveBayes(outcome ~ ., data = train_nb)
nb_pred <- predict(nb_model, newdata = test_nb)
nb_cm <- confusionMatrix(nb_pred, test_nb$outcome)
print(nb_cm)
nb_prob <- predict(nb_model, newdata = test_nb, type = "raw")[,2]
nb_roc <- roc(test$outcome, nb_prob)
plot(nb_roc, main = "ROC Curve - Naive Bayes", col = "purple")
print(auc(nb_roc))
cv_nb <- train(outcome ~ ., data = train_nb, method = "naive_bayes", metric = "ROC", trControl = ctrl_cv)


# --- Naive Bayes with tuning ---
tune_grid_nb <- expand.grid(usekernel = c(TRUE, FALSE), laplace = 0:2, adjust = 1)
cv_nb <- train(outcome ~ ., data = train_nb, method = "naive_bayes",
               trControl = ctrl_cv, tuneGrid = tune_grid_nb)
print(cv_nb$bestTune)
nb_train_pred <- predict(nb_model, newdata = train_nb)
nb_train_acc <- mean(nb_train_pred == train_nb$outcome)

# ----------------- KNN ------------------
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
df_knn <- df
df_knn[1:8] <- lapply(df_knn[1:8], normalize)
df_knn$outcome <- factor(df_knn$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
train_knn <- df_knn[index, ]
test_knn <- df_knn[-index, ]
knn_model <- train(outcome ~ ., data = train_knn, method = "knn", trControl = ctrl_cv, tuneLength = 10, metric = "ROC")
knn_class <- predict(knn_model, newdata = test_knn)
knn_cm <- confusionMatrix(knn_class, test_knn$outcome)
print(knn_cm)
knn_probs <- predict(knn_model, newdata = test_knn, type = "prob")[, "Diabetic"]
knn_roc <- roc(test_knn$outcome, knn_probs)
plot(knn_roc, main = "ROC Curve - KNN", col = "orange")
print(auc(knn_roc))
# --- KNN with hyperparameter tuning ---
tune_grid_knn <- expand.grid(k = seq(3, 25, 2))  # Tune odd k values
cv_knn <- train(outcome ~ ., data = train_knn, method = "knn",
                trControl = ctrl_cv, tuneGrid = tune_grid_knn)
print(cv_knn$bestTune)  # Best k
 
# --- Plotting KNN k vs Accuracy ---
ggplot(cv_knn$results, aes(x = k, y = Accuracy)) +
  geom_line() +
  geom_point() +
  labs(title = "KNN Accuracy vs. K", x = "K value", y = "Accuracy") +
  theme_minimal()
knn_train_pred <- predict(knn_model, newdata = train_knn)
knn_train_acc <- mean(knn_train_pred == train_knn$outcome)

# ----------------- DECISION TREE ------------------
tree_model <- rpart(outcome ~ ., data = train, method = "class")
rpart.plot(tree_model)
tree_pred <- predict(tree_model, newdata = test, type = "class")
tree_cm <- confusionMatrix(tree_pred, test$outcome)
print(tree_cm)
tree_probs <- predict(tree_model, newdata = test, type = "prob")[, "Diabetic"]
tree_roc <- roc(test$outcome, tree_probs)
plot(tree_roc, main = "ROC Curve - Decision Tree", col = "darkgreen")
print(auc(tree_roc))
train_tree <- train  # create a clean copy
train_tree$outcome <- factor(train_tree$outcome, levels = c("Non-diabetic", "Diabetic"),
                             labels = c("Non_diabetic", "Diabetic"))

cv_tree <- train(outcome ~ ., data = train_tree, method = "rpart", metric = "ROC", trControl = ctrl_cv)


# --- Decision Tree with hyperparameter tuning ---
tune_grid_tree <- expand.grid(cp = seq(0.001, 0.05, by = 0.005))  # Complexity parameter
cv_tree <- train(outcome ~ ., data = train_tree, method = "rpart",
                 trControl = ctrl_cv, tuneGrid = tune_grid_tree)
print(cv_tree$bestTune)  # Best cp

tree_train_pred <- predict(tree_model, newdata = train, type = "class")
tree_train_acc <- mean(tree_train_pred == train$outcome)

# ----------------- SVM ------------------
df_svm <- na.omit(df)  
df_svm$outcome <- factor(df_svm$outcome, levels = c("Non-diabetic", "Diabetic"),
                         labels = c("Non_diabetic", "Diabetic"))

# Train/test split
train_svm <- df_svm[index, ]
test_svm <- df_svm[-index, ]

# Train SVM model 
svm_model <- ksvm(outcome ~ ., data = train_svm, kernel = "rbfdot", prob.model = TRUE)
svm_pred <- predict(svm_model, newdata = test_svm)
svm_cm <- confusionMatrix(svm_pred, test_svm$outcome)
print(svm_cm)

# ROC and AUC
svm_probs <- predict(svm_model, newdata = test_svm, type = "probabilities")
svm_roc <- roc(test_svm$outcome, svm_probs[, "Diabetic"])
plot(svm_roc, main = "ROC Curve - SVM", col = "brown")
print(auc(svm_roc))

# Create a clean CV set with no NAs (in case df_svm still has some)
train_svm_cv <- na.omit(train_svm)

# Now train using caret with fixed labels
cv_svm <- train(outcome ~ ., data = train_svm_cv, method = "svmRadial",
                metric = "ROC", trControl = ctrl_cv)

# --- SVM with hyperparameter tuning ---
tune_grid_svm <- expand.grid(C = 2^(-2:2), sigma = 2^(-5:1))
cv_svm <- train(outcome ~ ., data = train_svm_cv, method = "svmRadial",
                trControl = ctrl_cv, tuneGrid = tune_grid_svm)
print(cv_svm$bestTune)  # Best (C, sigma)

svm_train_pred <- predict(cv_svm, newdata = train_svm)
svm_train_acc <- mean(svm_train_pred == train_svm$outcome)

# --------------------- Final Model Comparison (with Train Accuracy) ---------------------

# Ensure consistent outcome levels across train datasets
train2$outcome <- factor(train2$outcome, levels = c("Non_diabetic", "Diabetic"))
train_lda$outcome <- factor(train_lda$outcome, levels = c("Non_diabetic", "Diabetic"))
train_qda$outcome <- factor(train_qda$outcome, levels = c("Non_diabetic", "Diabetic"))
train_nb$outcome <- factor(train_nb$outcome, levels = c("Non_diabetic", "Diabetic"))
train_knn$outcome <- factor(train_knn$outcome, levels = c("Non_diabetic", "Diabetic"))
train_tree$outcome <- factor(train_tree$outcome, levels = c("Non_diabetic", "Diabetic"))
train_svm$outcome <- factor(train_svm$outcome, levels = c("Non_diabetic", "Diabetic"))

# Final results table
model_results <- data.frame(
  Model = c("Logistic Regression", "LDA", "QDA", "Naive Bayes", "KNN", "Decision Tree", "SVM"),
  Train_Accuracy = c(
    mean(log_train_pred == train2$outcome),
    mean(lda_train_pred == train_lda$outcome),
    mean(qda_train_pred == train_qda$outcome),
    mean(nb_train_pred == train_nb$outcome),
    mean(knn_train_pred == train_knn$outcome),
    mean(tree_train_pred == train$outcome),
    mean(svm_train_pred == train_svm$outcome)
  ),
  Test_Accuracy = c(
    log_cm$overall["Accuracy"],
    lda_cm$overall["Accuracy"],
    qda_cm$overall["Accuracy"],
    nb_cm$overall["Accuracy"],
    knn_cm$overall["Accuracy"],
    tree_cm$overall["Accuracy"],
    svm_cm$overall["Accuracy"]
  ),
  Test_AUC = c(
    auc(log_roc),
    auc(lda_roc),
    auc(qda_roc),
    auc(nb_roc),
    auc(knn_roc),
    auc(tree_roc),
    auc(svm_roc)
  ),
  CV_Accuracy = c(
    get_metric(cv_log, "Accuracy"),
    get_metric(cv_lda, "Accuracy"),
    get_metric(cv_qda, "Accuracy"),
    get_metric(cv_nb, "Accuracy"),
    get_metric(knn_model, "Accuracy"),
    get_metric(cv_tree, "Accuracy"),
    get_metric(cv_svm, "Accuracy")
  ),
  CV_AUC = cv_auc_list
)

# Print final table
print(model_results)


# ---------------------- MODEL COMPARISON PLOT ----------------------

# Long format for ggplot
model_results_long <- model_results %>%
  dplyr::select(Model, CV_Accuracy, CV_AUC) %>%
  pivot_longer(cols = c("CV_Accuracy", "CV_AUC"),
               names_to = "Metric", values_to = "Score")

# Plot
ggplot(model_results_long, aes(x = reorder(Model, Score), y = Score, fill = Metric)) +
  geom_col(position = "dodge") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Model Comparison: Cross-Validated Accuracy vs AUC",
       x = "Model", y = "Score") +
  theme(legend.position = "top",
        text = element_text(size = 12))

# ----------------------------- Main Effects Model Training -----------------------------

set.seed(123)

#Re-split data
index <- createDataPartition(df$outcome, p = 0.7, list = FALSE)
train_main <- df[index, ]
test_main  <- df[-index, ]

#model evaluation
library(pROC)
evaluate_model <- function(model, train_data, test_data, pred_type = "response", prob_column = "Diabetic") {
  if (inherits(model, "lda") || inherits(model, "qda")) {
    # Special handling for LDA/QDA
    pred_obj <- predict(model, newdata = test_data)
    test_pred <- pred_obj$class
    test_probs <- pred_obj$posterior[, "Diabetic"]
    
    train_pred <- predict(model, newdata = train_data)$class
  } else if (pred_type == "response") {
    test_probs <- predict(model, newdata = test_data, type = "response")
    test_pred <- ifelse(test_probs > 0.5, "Diabetic", "Non-diabetic")
    
    train_pred <- ifelse(predict(model, newdata = train_data, type = "response") > 0.5, "Diabetic", "Non-diabetic")
  } else if (pred_type == "class") {
    test_pred <- predict(model, newdata = test_data)
    test_probs <- predict(model, newdata = test_data, type = "prob")[, prob_column]
    
    train_pred <- predict(model, newdata = train_data)
  } else if (pred_type == "prob") {
    test_probs <- predict(model, newdata = test_data, type = "prob")[, prob_column]
    test_pred <- predict(model, newdata = test_data)
    
    train_pred <- predict(model, newdata = train_data)
  } else if (pred_type == "probabilities") {
    test_probs <- attr(predict(model, newdata = test_data, type = "probabilities"), "probabilities")[, "Diabetic"]
    test_pred <- predict(model, newdata = test_data)
  } else if (pred_type == "raw") {
    test_probs <- predict(model, newdata = test_data, type = "raw")[, "Diabetic"]
    test_pred <- predict(model, newdata = test_data, type = "class")
    
    train_pred <- predict(model, newdata = train_data, type = "class")
  }

  
  test_cm <- caret::confusionMatrix(factor(test_pred, levels = c("Non-diabetic", "Diabetic")), test_data$outcome)
  test_auc <- pROC::auc(test_data$outcome, test_probs)
  train_acc <- mean(train_pred == train_data$outcome)
  
  list(train_acc = train_acc, test_acc = test_cm$overall["Accuracy"], test_auc = test_auc)
}

# Logistic Regression
log_model_main <- glm(outcome ~ ., data = train_main, family = "binomial")
log_results <- evaluate_model(log_model_main, train_main, test_main, "response")

# LDA
lda_model_main <- MASS::lda(outcome ~ ., data = train_main)
lda_results <- evaluate_model(lda_model_main, train_main, test_main, "class")

# QDA
qda_model_main <- MASS::qda(outcome ~ ., data = train_main)
qda_results <- evaluate_model(qda_model_main, train_main, test_main, "class")

# Naive Bayes
nb_model_main <- e1071::naiveBayes(outcome ~ ., data = train_main)
nb_results <- evaluate_model(nb_model_main, train_main, test_main, "raw")

# KNN (normalize first)
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
df_knn <- df
df_knn[1:8] <- lapply(df_knn[1:8], normalize)
df_knn$outcome <- factor(df_knn$outcome, levels = c("Non-diabetic", "Diabetic"), labels = c("Non_diabetic", "Diabetic"))
train_knn_main <- df_knn[index, ]
test_knn_main <- df_knn[-index, ]

knn_model_main <- caret::train(outcome ~ ., data = train_knn_main, method = "knn", tuneLength = 5)
knn_results <- evaluate_model(knn_model_main, train_knn_main, test_knn_main, "prob")

# Decision Tree
tree_model_main <- rpart::rpart(outcome ~ ., data = train_main, method = "class")
tree_probs <- predict(tree_model_main, newdata = test_main, type = "prob")[, "Diabetic"]
tree_pred <- ifelse(tree_probs > 0.5, "Diabetic", "Non-diabetic")
tree_cm <- caret::confusionMatrix(factor(tree_pred, levels = c("Non-diabetic", "Diabetic")), test_main$outcome)
tree_auc <- pROC::auc(test_main$outcome, tree_probs)

# Calculate training accuracy manually
tree_train_pred <- predict(tree_model_main, newdata = train_main, type = "class")
tree_train_acc <- mean(tree_train_pred == train_main$outcome)

# Store as a list
tree_results <- list(train_acc = tree_train_acc, test_acc = tree_cm$overall["Accuracy"], test_auc = tree_auc)

# SVM 
# Fix outcome levels and NA removal
df_svm_main <- na.omit(df)
df_svm_main$outcome <- factor(df_svm_main$outcome, levels = c("Non-diabetic", "Diabetic"), 
                              labels = make.names(c("Non-diabetic", "Diabetic")))

# Train/test split
train_svm_main <- df_svm_main[index, ]
test_svm_main  <- df_svm_main[-index, ]

# Train SVM model using caret
ctrl_svm <- trainControl(
  method = "none",  # no CV here — just train on main effects
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

svm_model_main <- train(outcome ~ ., data = train_svm_main,
                        method = "svmRadial",
                        trControl = ctrl_svm,
                        metric = "ROC",
                        preProcess = c("center", "scale"))

# Predict on test set
svm_probs <- predict(svm_model_main, newdata = test_svm_main, type = "prob")[, "Diabetic"]
svm_pred <- predict(svm_model_main, newdata = test_svm_main)

# Evaluation
svm_cm <- caret::confusionMatrix(svm_pred, test_svm_main$outcome)
svm_auc <- pROC::auc(test_svm_main$outcome, svm_probs)

# Training accuracy
svm_train_pred <- predict(svm_model_main, newdata = train_svm_main)
svm_train_acc <- mean(svm_train_pred == train_svm_main$outcome)

# Final result list
svm_results <- list(train_acc = svm_train_acc, test_acc = svm_cm$overall["Accuracy"], test_auc = svm_auc)
main_effects_results <- data.frame(
  Model = c("Logistic Regression", "LDA", "QDA", "Naive Bayes", "KNN", "Decision Tree", "SVM"),
  Train_Accuracy = c(
    log_results$train_acc,
    lda_results$train_acc,
    qda_results$train_acc,
    nb_results$train_acc,
    knn_results$train_acc,
    tree_results$train_acc,
    svm_results$train_acc
  ),
  Test_Accuracy = c(
    log_results$test_acc,
    lda_results$test_acc,
    qda_results$test_acc,
    nb_results$test_acc,
    knn_results$test_acc,
    tree_results$test_acc,
    svm_results$test_acc
  ),
  Test_AUC = c(
    log_results$test_auc,
    lda_results$test_auc,
    qda_results$test_auc,
    nb_results$test_auc,
    knn_results$test_auc,
    tree_results$test_auc,
    svm_results$test_auc
  )
)

# Print the final table
print(main_effects_results)