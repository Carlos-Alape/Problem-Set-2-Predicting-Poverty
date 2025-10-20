setwd("C:/Users/p.rinconr/Documents/Set2BDML/Bases")

test_hogares   <- read.csv("test_hogares.csv", encoding = "UTF-8")
test_personas  <- read.csv("test_personas.csv", encoding = "UTF-8")

install.packages("tidyverse")
library(tidyverse)
library(dplyr)

# BASES DE DATOS DE TESTING