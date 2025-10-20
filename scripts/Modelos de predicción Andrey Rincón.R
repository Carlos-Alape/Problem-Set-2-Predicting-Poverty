if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret, glmnet, MLmetrics, dplyr, DMwR, rpart, rpart.plot, Metrics, pROC)

set.seed(2025)
setwd('C:/Users/investigacion/Desktop/Set2BDML/Bases')

# EstableceMOS directorio 
setwd('C:/Users/investigacion/Desktop/Set2BDML/Bases')


# Cargar datos
train <- read.csv("train_unido.csv", encoding = "UTF-8")
test <- read.csv("test_unido.csv", encoding = "UTF-8")

# Convertir variable pobre a factor
train$pobre <- factor(train$pobre,
                      levels = c("0", "1"),
                      labels = c("No", "Yes"))

# Convertir variables categóricas a factores
factor_vars <- c("mujer", "jefe_hogar", "amo_casa", "inactivo", "ocupado", 
                 "estudiante", "adulto_mayor", "desempleado", "busca_trabajo", 
                 "primaria", "secundaria", "media", "superior", "rural", 
                 "menores_edad")



for (v in factor_vars) {
  train[[v]] <- as.factor(train[[v]])
  test[[v]] <- factor(test[[v]], levels = levels(train[[v]]))
}



# MODELO ELASTIC NET 

fiveStats <- function(...) {
  c(twoClassSummary(...), defaultSummary(...))
}


tune_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.2),  
  lambda = exp(seq(log(1e-6),   
                   log(0.1),   
                   length.out = 25))
)


ctrl <- trainControl(
  method = "cv",
  number = 5,           
  classProbs = TRUE,
  summaryFunction = fiveStats,
  sampling = "up",     
  allowParallel = TRUE,
  savePredictions = "final"
)

# Entrenamos el modelo 
set.seed(2025)
model_elastic_fixed <- train(
  pobre ~ nper + busca_trabajo + desempleado + amo_casa + 
    inactivo + ocupado + primaria + secundaria + media + edad + 
    menores_edad + rural + mujer + estudiante + 
    adulto_mayor + superior,
  data = train,
  method = "glmnet",
  family = "binomial",
  metric = "F",
  trControl = ctrl,
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)

print(model_elastic_fixed$bestTune)



# Probamos predicciones en test
pred_fixed <- predict(model_elastic_fixed, newdata = test, type = "raw")
cat("\nDistribución de predicciones corregidas:\n")
print(table(pred_fixed))

prop_pobre_pred <- mean(pred_fixed == "Yes")
prop_pobre_pred



# Transformamos variable a 0 Y 1 
submission <- data.frame(
  id = test$id,
  pobre = ifelse(pred_fixed == "Yes", 1, 0)
)


best_result <- model_elastic_fixed$results[
  model_elastic_fixed$results$alpha == model_elastic_fixed$bestTune$alpha &
    model_elastic_fixed$results$lambda == model_elastic_fixed$bestTune$lambda, 
]
best_result

roc_obj_train <- roc(train$pobre, predict(model_elastic_fixed, newdata = train, type = "prob")[, "Yes"], levels = c("No", "Yes"), direction = "<")
plot(roc_obj_train, main = "Curva ROC - Elastic Net Fixed  (Entrenamiento)", col = "blue")
text(x = 0.5, y = 0.4, labels = paste("AUC =", round(auc_train, 4)), cex = 1.5)
auc_train <- auc(roc_obj_train)
cat("AUC en entrenamiento:", auc_train, "\n")


train_pred <- predict(model_elastic_fixed, newdata = train, type = "raw")
f1_train <- F1_Score(y_true = train$pobre, y_pred = train_pred, positive = "Yes")
cat("F1 score en entrenamiento:", f1_train, "\n")


# Creamos archivo para subir
lambda_str <- gsub("\\.", "_", sprintf("%.6f", model_elastic_fixed$bestTune$lambda))
alpha_str  <- gsub("\\.", "_", sprintf("%.2f", model_elastic_fixed$bestTune$alpha))





filename <- paste0(
  "ElasticNet_alpha", alpha_str,
  "_lambda", lambda_str,
  "_F1_", f1_str,
  ".csv"
)

write.csv(submission, filename, row.names = FALSE)
prop.table(table(submission$pobre))



#  MODELO LOGIT 

# Establecemos la grilla 
lambda_grid <- 10^seq(-4, 0.01, length = 10) 

# Construimos modelo de predicción
ridge<- train(
  pobre ~ nper + busca_trabajo + desempleado + amo_casa + 
    inactivo + ocupado + primaria + secundaria + media + edad + 
    menores_edad + rural + mujer + estudiante + 
    adulto_mayor + superior,
  data = train, 
  method = "glmnet",
  family = "binomial", 
  metric = "Accuracy",
  trControl = ctrl,
  tuneGrid = expand.grid(alpha = 1,lambda=lambda_grid), 
  preProcess = c("center", "scale")
)

#Aplicamos el modelos a testinbg 
test$pobre <- predict(ridge, test)


pred_logit <- predict(ridge, newdata = test, type = "raw")


#Cambiamos  las categorías a 0 y 1. 
test<- test%>% 
  mutate(pobre =ifelse(pobre=="Yes",1,0))
head(test %>% select(id,pobre))


# Creamos el archivo de subida a kaggle 
submit <- test %>% select(id, pobre)
filename <- "submission_logitrigde.csv"

# Guardar el archivo
write.csv(submit, filename, row.names = FALSE)
prop.table(table(submit$pobre))



# ARBOLES

fiveStats <- function(...) {
  c(twoClassSummary(...), defaultSummary(...))
}

ctrl <- trainControl(
  method = "cv",
  number = 10,  
  summaryFunction = fiveStats,
  classProbs = TRUE,
  verboseIter = TRUE,  
  savePredictions = "final",  
  sampling = "down" 
)

grid <- expand.grid(cp = seq(0, 0.05, by = 0.005))

# Modelo 
set.seed(2025)
modelo_cart <- train(
  pobre ~ nper + busca_trabajo + desempleado + amo_casa + 
    inactivo + ocupado + primaria + secundaria + media + edad + 
    menores_edad + rural + mujer + estudiante + adulto_mayor + superior,
  data = train,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = grid
)

print(modelo_cart)

modelo_cart$bestTune

# Metricas 

# Extraer resultados del mejor modelo
best_results <- modelo_cart$results[which.max(modelo_cart$results$ROC), ]
best_results

# Metricas en conjunto de entrenamiento

# Predicciones en train
pred_train_clase <- predict(modelo_cart, newdata = train, type = "raw")
pred_train_prob <- predict(modelo_cart, newdata = train, type = "prob")

# Matriz de confusión
conf_matrix_train <- confusionMatrix(pred_train_clase, train$pobre, positive = "Yes")
print(conf_matrix_train)

# ROC y AUC en entrenbamiento 
train$pobre_num <- as.numeric(train$pobre == "Yes")
roc_train <- roc(train$pobre_num, pred_train_prob$Yes)
auc_train <- auc(roc_train)

# curva ROC
plot(roc_train, main = "Curva ROC - Modelo Árbol (Entrenamiento)", 
     col = "blue", lwd = 2)
text(0.5, 0.3, paste("AUC =", round(auc_train, 4)), cex = 1.2)

# Convertir predicciones a formato 0 y 1

test_submission <- test %>%
  mutate(pobre = ifelse(predicciones_clase == "Yes", 1, 0))

# Crear archivo de subida a Kaggle
submit <- test_submission %>% select(id, pobre)
filename <- "Arbol.csv"
write.csv(submit, filename, row.names = FALSE)


prop.table(table(submit$pobre))



#BAGGING

for (v in factor_vars) {
  train[[v]] <- as.factor(train[[v]])
  test[[v]] <- factor(test[[v]], levels = levels(train[[v]]))
}

# Control de validación cruzada con sampling
ctrl <- trainControl(
  method = "cv",
  number = 5,          
  classProbs = TRUE,
  verboseIter = TRUE,
  savePredictions = "final",
  sampling = "down",
  allowParallel = TRUE  
)

set.seed(2025)
modelo_bagging_caret <- train(
  pobre ~ nper + busca_trabajo + desempleado + amo_casa + 
    inactivo + ocupado + primaria + secundaria + media + edad + 
    menores_edad + rural + mujer + estudiante + adulto_mayor + superior,
  data = train,
  method = "treebag",
  trControl = ctrl,
  metric = "ROC",
  nbagg = 100,          
  control = rpart.control(
    minsplit = 50,              
    minbucket = 20,   
    cp = 0.001,        
    maxdepth = 15     
  )
)

print(modelo_bagging_caret)

# Aplicamos predicción a testing 
predicciones_clase_caret <- predict(modelo_bagging_caret, newdata = test)
predicciones_prob_caret <- predict(modelo_bagging_caret, newdata = test, type = "prob")

# Convertir a 0/1 la variable pobre
test_submission_caret <- test %>%
  mutate(pobre = ifelse(predicciones_clase_caret == "Yes", 1, 0))

# Crear submission
submit_caret <- test_submission_caret %>% select(id, pobre)
write.csv(submit_caret, "bagging_caret.csv", row.names = FALSE)

prop.table(table(submit_caret$pobre))

