#import data sous format csv
mydata=read.csv("C:\\Users\\lenovo\\Desktop\\R project\\Obesity.csv")
# Ouvrir la visualisation des données "mydata"
View(mydata)

# Attacher les données "mydata" pour accéder plus facilement à ses variables
attach(mydata)

# Tableau de fréquences des variables "family_history_with_overweight" et "Gender"
table(family_history_with_overweight, Gender)

# Résumé statistique des variables de "mydata"
summary(mydata)

# Structure des données "mydata"
str(mydata)

#Welch Anova 
library(car)
model <- aov(Weight~Gender*family_history_with_overweight, data = mydata)
results <- Anova(model, type = "III", white.adjust = TRUE)
results

#levene test
leveneTest(mydata$Weight~mydata$Gender*mydata$family_history_with_overweight)

#afficher les groupes des données 
grouped_data <- split(mydata, list(Gender,family_history_with_overweight))
for (i in 1:length(grouped_data)) {
  print(names(grouped_data)[i])
}

# Boxplot avec les moyennes
ggplot(mydata, aes(x = Gender, y = mydata$Weight, fill = factor(family_history_with_overweight))) +
  geom_boxplot() +
  stat_summary(fun = "mean", geom = "point", shape = 18, size = 5, color = "red", position = position_dodge(width = 0.75)) +
  labs(x = "Gender", y = "Weight", fill = "Family history with overweight")


# Calcul des moyennes de poids par genre
means <- round(tapply(mydata$Weight, mydata$Gender, mean), digits=2)

# Calcul des moyennes de poids par genre et antécédents familiaux de surpoids
moyennes <- aggregate(Weight ~ Gender + family_history_with_overweight, data = mydata, FUN = mean)

# Analyse de variance (ANOVA) pour tester la significativité des différences de poids selon le genre et les antécédents familiaux de surpoids
anova <- aov(mydata$Weight ~ mydata$Gender * mydata$family_history_with_overweight, data = mydata)
summary(anova)

# Tableau croisé des fréquences pour les variables Gender et family_history_with_overweight
table(mydata$Gender, mydata$family_history_with_overweight)

# Test de normalité de Shapiro-Wilk sur les résidus de l'ANOVA
shapiro.test(resid(anova))

# Test de comparaisons multiples de Tukey pour déterminer les différences significatives entre les moyennes des groupes dans l'ANOVA
res <- TukeyHSD(anova)

# Installer la librairie "emmeans"
install.packages('emmeans')

# Charger la librairie "emmeans"
library(emmeans)

# Obtenir les moyennes des facteurs "Gender" pour chaque niveau de "family_history_with_overweight"
res.g <- emmeans(anova, ~Gender | family_history_with_overweight)

# Obtenir les moyennes des facteurs "family_history_with_overweight" pour chaque niveau de "Gender"
res.f <- emmeans(anova, ~family_history_with_overweight | Gender)

# Afficher les résultats des moyennes des facteurs "Gender"
res.g

# Afficher les résultats des moyennes des facteurs "family_history_with_overweight"
res.f

# Tracer un graphique des moyennes des facteurs "family_history_with_overweight"
plot(res.f)

# Tracer un graphique des moyennes des facteurs "Gender"
plot(res.g)




