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