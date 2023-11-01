# Importer les données
data <- read.csv("C:\\Users\\lenovo\\Desktop\\R project\\Obesity.csv")

# Encodage des variables qualitatives en variables binaires
Gender <- ifelse(data$Gender == "F", 0, 1)
data$family_history_with_overweight <- ifelse(data$family_history_with_overweight == "no", 0, 1)
shapiro.test(data$Weight)
ks.test(data$Weight, "pnorm", mean(data$Weight), sd(data$Weight))

# Créer un modèle de régression linéaire simple
modele <- lm(data$Weight ~ data$Gender + data$family_history_with_overweight, data = data)

# Vérification des hypothèses de normalité, d'homoscédasticité et d'indépendance des résidus
plot(modele)

# Interpréter les résultats de la régression linéaire
summary(modele)