# Rossmann-Store-Sales

Drugstore Sales Forecast with Machine Learning.

![Image](img/README_banner.png)

*This Data Science project presents a sales prediction for one of the largest drugstore chains in Europe, called Rossmann. It is a German company with more than 3,000 stores located across seven European countries. The project was inspired in a database available from a [Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales/overview).*

Data Science is an extremely diverse field: it envolves programming, mathematics and statistics, data analysis and visualization, machine learning... 
Each subject needs to be studied deeply and individually, although in practise they are strongly related in a step-by-step resolution.

Developing an end-to-end project is an opportunity to apply skills from the different existent branches of DS, from business understanding and data description to model deployment. This led me to work carefully on each step of this project for having a broad and extensive understanding on each of the steps followed through coding.

Here is what I will cover:

- 1.[ Business Problem](#business-problem)
- 2.[ Data Description](#data-description)
- 3.[ Exploratory Data Analysis](#exploratory-data-analysis)
- 4.[ Machine Learning Modelling](#machine-learning-modelling)
- 5.[ Project Outcomes](#project-outcomes)
- 6.[ Conclusion](#conclusion)
- [Special Thanks](special-thanks)
- [References](#references)

Moreover, you can check the (so far) final version of the Jupyter Notebook coding [here](v10-Rossmann_Store_Sales.ipynb), as well as the slide presentation created for Business Storytelling [here](Project_Storytelling.pdf).

## Business Problem

Imagine you receive a specific task from your boss telling you the exact results that the company needs for the moment. You spend valuable working hours trying to develop the most fittable solution and acquire accurate results, and when you are finally ready to present them to your superiors it turns out that those results are not enough for giving the company the best insights in order to solve their main issues.

This is where business understanding goes in. Before starting any data science project, understanding the reason why this project is going to be developed in the first place is indispensable, not only for arriving into the best solution but also for checking if all the data needed for your project is available or can be accessed.

For this project, in order to motivate the development of the challenge's solution, a fictional situation was created. The business problem is:

- The sales forecast was requested by the CFO at a monthly meeting on store results;
  - Root cause: difficulty in determining the investment value for the renovations of each store.

Therefore, **each store daily sales for the upcoming six weeks** are going to be predicted, giving the CFO a better understanding of the individual incomes on the next days and allowing him to invest according to the profit of the stores.

### General overview of the problem

- Granularity: forecast sales per day, per store, for the next 42 days (6 weeks)
- Problem Type: sales forecast
- Potential ML Methods: Time Series, Regression
- Delivery Format: the total sales value of each store at the end of 6 weeks

## Data Description

The used data consists of specific information regarding sales data for 1,115 Rossmann stores, acquired between **2013-01-01** and **2015-07-31**. The features included are:
- Date
  - Date of sales, holidays
- Store
  - Type, assortment
- Costumer
  - Quantity
- Competition
  - Distance, open since
- Promotion
  - Period of promo, if there was a consecutive promo
  


### Feature Engineering

## Exploratory Data Analysis

## Machine Learning Modelling
- Regression
- Time Series

## Project Outcomes

### Business Performance

### ML Performance

## Model Deployment

## Conclusion

## Special Thanks

I would like to give a special thanks to Meigarom Lopes for providing the orientation needed for me to achieve these results, as well as for improving my Data Science knowledge throughout the course Data Science in Production. For more details, please check his course (in Portuguese) [here](https://sejaumdatascientist.com/como-ser-um-data-scientist/).

## References
