PracticalMachineLearning_Project
================================

Coursera Repository - Needed for Project Course - Practical Machine Learning


---
title: "Practical Machine Learning - Project"
author: "Jaime Laorden"
date: "Thursday, December 18, 2014"
output: html_document
---

<!-- # Databases source:: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recogniti#on of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation #with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3MGRd7WU2 -->

#### Tittle: Activity Recognition of Weight Lifting Exercises

### Summary
This report presents the proccess followed to generate a CART rpart Clssification model for Activity Recognition of Weight Lifting Exercises, with final results obteined after apllying the model into out-sample data set. Cros Validation was use to build diferent Decission trees and lower xvalidation error rate tree was selected.

*Databases source:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recogniti#on of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation #with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3MGRd7WU2


Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 
