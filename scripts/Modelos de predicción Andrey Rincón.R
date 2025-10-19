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

# Creamos archivo para subir
lambda_str <- gsub("\\.", "_", sprintf("%.6f", model_elastic_fixed$bestTune$lambda))
alpha_str  <- gsub("\\.", "_", sprintf("%.2f", model_elastic_fixed$bestTune$alpha))


best_result <- model_elastic_fixed$results[
  model_elastic_fixed$results$alpha == model_elastic_fixed$bestTune$alpha &
    model_elastic_fixed$results$lambda == model_elastic_fixed$bestTune$lambda, 
]
f1_str <- gsub("\\.", "_", sprintf("%.4f", best_result$F))

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

# Construimos nuestro modelo 

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
  sampling = "down" # balancea las clases, puedes probar "up" o "smote"
)

grid <- expand.grid(cp = seq(0, 0.05, by = 0.005))

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

#Aplicamos a testing
predicciones_clase <- predict(modelo_cart, newdata = test, type = "raw")
predicciones_prob <- predict(modelo_cart, newdata = test, type = "prob")

# Convertir predicciones a formato 0 y 1

test_submission <- test %>%
  mutate(pobre = ifelse(predicciones_clase == "Yes", 1, 0))

# Crear archivo de subida a Kaggle
submit <- test_submission %>% select(id, pobre)
filename <- "Arbol.csv"
write.csv(submit, filename, row.names = FALSE)


prop.table(table(submit$pobre))

